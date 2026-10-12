"""Experiment reports (evaluator.protocols.experiments): the statistics, and an RL experiment trained and reported.
`jobs/tests/test_golden_eval.py` holds every report to the numbers the old experiment scripts produced."""

import arena.snake
import pytest
from jobcore.specs import ExperimentSpec
from telemetry import open_job_store, open_stores
from trainer import train
from trainer.algorithms import reinforcement

from evaluator import evaluate
from evaluator.protocols import snake_score
from evaluator.protocols.experiments import across_seeds, paired_permutation_p


def test_paired_permutation_p_is_exact():
    assert paired_permutation_p([1, 2, 3], [1, 2, 3]) == 1.0
    # five pairs all in one direction: only the all-positive and all-negative sign patterns are as extreme
    assert paired_permutation_p([5, 6, 7, 8, 9], [1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    with pytest.raises(ValueError):
        paired_permutation_p([1, 2], [1])


def test_across_seeds_is_the_regime_level_view():
    agree = across_seeds({s: {"elo": e} for s, e in enumerate([103.0, 124.0, 127.0, 120.0, 126.0])})
    assert agree["positive"] == 5 and agree["sign_flip_p"] == 0.0625  # the floor with five seeds
    split = across_seeds({s: {"elo": e} for s, e in enumerate([24.0, 21.0, -7.0, 22.0, -33.0])})
    assert split["positive"] == 3 and split["sign_flip_p"] > 0.5


def test_an_rl_experiment_reports_each_arm_and_pairs_it_with_its_baseline(monkeypatch):
    monkeypatch.setattr(reinforcement, "MONITOR_SEEDS", (20_000, 20_001))
    monkeypatch.setattr(arena.snake, "HELD_OUT_SEEDS", tuple(range(10_000, 10_010)))
    spec = ExperimentSpec.model_validate(
        {
            "name": "t",
            "seeds": [0, 1],
            "defaults": {
                "game": "snake",
                "budget": {"iterations": 2},
                "params": {"steps_per_iteration": 5_000, "agent": {"epsilon_decay_steps": 5_000}},
            },
            "arms": {
                "q-learning": {"train": {"algorithm": "rl.q_learning"}},
                "sarsa": {"train": {"algorithm": "rl.sarsa"}, "baseline": "q-learning"},
            },
            "report": {"protocol": "experiment.rl-report"},
        }
    )
    open_job_store().put_experiment(spec.name, spec.model_dump(mode="json"))
    for _arm, _seed, train_spec in spec.expand():
        train(train_spec)

    report = evaluate(spec.report_spec())
    assert set(report["arms"]) == {"q-learning", "sarsa"}
    sarsa = report["arms"]["sarsa"]
    versus = sarsa["vs_baseline"]
    assert sarsa["n"] == 2 and versus["arm"] == "q-learning" and versus["pairs"] == 2 and 0 < versus["paired_p"] <= 1
    assert "vs_baseline" not in report["arms"]["q-learning"]
    run = report["runs"][0]
    assert run["env_steps"] == 10_000 and run["parameters"] == 2048 * 3 and run["visited_states"] > 0
    assert run["curve"] and run["curve"][-1][0] == 10_000

    # the champion is an ordinary leaderboard entrant only outside an experiment: these stay aggregated here
    stores = open_stores()
    assert snake_score.champion_entrants(stores.registry, stores.metrics, stores.artifacts, "snake") == []
