import evaluate
import pytest
import rl_experiment
import rl_run
import run_context
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry


def test_paired_permutation_p_is_exact():
    assert rl_experiment.paired_permutation_p([1, 2, 3], [1, 2, 3]) == 1.0
    # five pairs all in one direction: only the all-positive and all-negative sign patterns are as extreme
    assert rl_experiment.paired_permutation_p([5, 6, 7, 8, 9], [1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    with pytest.raises(ValueError):
        rl_experiment.paired_permutation_p([1, 2], [1])


def test_an_experiment_runs_arms_by_seed_skips_what_is_done_and_reports_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(rl_experiment, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(rl_experiment, "STEPS_PER_ITERATION", 5_000)
    monkeypatch.setattr(rl_run, "MONITOR_SEEDS", (20_000, 20_001))
    monkeypatch.setattr(evaluate, "HELD_OUT_SEEDS", tuple(range(10_000, 10_010)))
    small = {
        "q-learning": rl_experiment.Arm("q_learning", {"epsilon_decay_steps": 5_000}, steps=10_000, baseline=None),
        "sarsa": rl_experiment.Arm("sarsa", {"epsilon_decay_steps": 5_000}, steps=10_000),
    }
    monkeypatch.setattr(rl_experiment, "ARMS", small)

    rl_experiment.main(["run", "--name", "t", "--arms", "q-learning,sarsa", "--seeds", "0-1"])
    rl_experiment.main(["run", "--name", "t", "--arms", "q-learning", "--seeds", "0"])  # all done: skipped
    runs = SqliteRunRegistry(tmp_path / "runs.db").list_runs()
    assert len(runs) == 4 and {r.config["arm"] for r in runs} == {"q-learning", "sarsa"}
    assert all(r.config["experiment"] == "t" and r.config["representation"] in ("q_learning", "sarsa") for r in runs)

    report = rl_experiment.build_report("t")
    assert set(report["arms"]) == {"q-learning", "sarsa"}
    sarsa = report["arms"]["sarsa"]
    versus = sarsa["vs_baseline"]
    assert sarsa["n"] == 2 and versus["arm"] == "q-learning" and versus["pairs"] == 2 and 0 < versus["paired_p"] <= 1
    assert "vs_baseline" not in report["arms"]["q-learning"]
    run = report["runs"][0]
    assert run["env_steps"] == 10_000 and run["parameters"] == 2048 * 3 and run["visited_states"] > 0
    assert run["curve"] and run["curve"][-1][0] == 10_000

    # the champion is an ordinary leaderboard entrant only outside an experiment: these stay aggregated here
    registry, metrics, artifacts = (
        SqliteRunRegistry(tmp_path / "runs.db"),
        FileMetricsStore(tmp_path / "metrics"),
        FileArtifactStore(tmp_path / "artifacts"),
    )
    assert evaluate.champion_entrants(registry, metrics, artifacts, "snake") == []
