import random

import checkers_pbt_run as pbt
import rl
import run_context
from games.checkers_openings import ballot
from telemetry import FileMetricsStore, SqliteRunRegistry


def test_settings_are_sampled_in_range_and_perturbed_within_it():
    rng = random.Random(0)
    for _ in range(50):
        s = pbt.sample_settings(rng)
        assert 3e-4 <= s["learning_rate"] <= 3e-3 and 0.5 <= s["lambda"] <= 0.9 and s["pfsp"] in (0.0, 2.0)
        p = pbt.perturb({**s, "lambda": 0.95, "pool_fraction": 0.9}, rng)
        assert p["lambda"] <= 1.0 and 0.05 <= p["pool_fraction"] <= 0.95
        assert min(abs(p["learning_rate"] / s["learning_rate"] - f) for f in (0.8, 1.25)) < 1e-12


def test_the_round_robin_ranks_a_trained_member_above_an_untrained_one():
    trained = rl.CheckersSelfPlay(1, params={"hidden": 8})
    trained.train(2000)
    fresh = rl.CheckersSelfPlay(2, params={"hidden": 8})
    scores = pbt.round_robin([trained.snapshot(), fresh.snapshot()], list(ballot()[:3]), depth=1, seed_base=0)
    assert abs(sum(scores) - 1.0) < 1e-9  # two players: their points per game sum to 1
    assert scores[0] > scores[1]


def test_a_tiny_population_runs_exploits_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(pbt, "BASE", {"hidden": 4, "pool_every": 20, "pool_size": 2})
    run_id = pbt.main(members=4, games=40, interval=20, depth=1, openings_per_pair=1, rng_seed=3)
    history = FileMetricsStore(tmp_path / "metrics").history(run_id)
    assert len(history) == 2
    assert history[0].extras["replaced"] == 1.0  # a quarter of four, after the first round robin
    assert history[-1].extras["replaced"] == 0.0  # nothing is copied after the last one
    assert history[0].best_fitness >= history[0].mean_fitness >= history[0].worst_fitness
    import json

    from telemetry import FileArtifactStore

    log = json.loads(FileArtifactStore(tmp_path / "artifacts").get_program(f"{run_id}-population"))
    assert [len(g["scores"]) for g in log] == [4, 4] and len(log[0]["copied"]) == 1
    run = SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id)
    assert run.status == "completed" and run.config["population"] == 4 and run.summary["cost"]["episodes"] == 160
