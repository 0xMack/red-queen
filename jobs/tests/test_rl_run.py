import math

import evaluate
import pytest
import rl_run
import run_context
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry


def test_an_rl_run_records_iterations_like_any_other_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(rl_run, "MONITOR_SEEDS", (20_000, 20_001, 20_002))

    run_id = rl_run.main(algorithm="random", iterations=3, steps_per_iteration=2000, held_out_every=2, rng_seed=4)

    registry, metrics = SqliteRunRegistry(tmp_path / "runs.db"), FileMetricsStore(tmp_path / "metrics")
    run = registry.get_run(run_id)
    assert run.status == "completed"
    assert run.config["representation"] == "random" and run.config["paradigm"] == "reinforcement_learning"
    assert (
        run.config["interface"] == "snake/features.v1+relative3.v1" and run.config["action_space"]["kind"] == "discrete"
    )
    history = metrics.history(run_id)
    assert [h.generation for h in history] == [0, 1, 2]
    assert [h.held_out_score is not None for h in history] == [True, False, True]  # every 2nd + the last
    assert all(math.isclose(h.diversity, math.log(3)) for h in history)
    assert all(h.worst_fitness <= h.mean_fitness <= h.best_fitness for h in history)
    assert [h.extras["env_steps"] for h in history] == [2000.0, 4000.0, 6000.0]
    assert run.summary["cost"]["env_steps"] == 6000 and run.summary["cost"]["episodes"] > 0
    snapshot = FileArtifactStore(tmp_path / "artifacts").get_program(history[-1].champion_ref)
    assert snapshot == b'{"type": "random", "num_actions": 3}'

    # Not a leaderboard entrant until modelpack can load RL champions (docs/design/0010 Phase 1).
    entrants = evaluate.champion_entrants(registry, metrics, FileArtifactStore(tmp_path / "artifacts"), "snake")
    assert entrants == []


def test_reach1d_runs_have_no_interface_or_held_out_score(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    run_id = rl_run.main(env_id="reach1d", iterations=2, steps_per_iteration=1000)
    history = FileMetricsStore(tmp_path / "metrics").history(run_id)
    run = SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id)
    assert run.config["game"] == "reach1d" and run.config["interface"] is None
    assert all(h.held_out_score is None for h in history)
    assert run.config["action_space"] == {"kind": "continuous", "values": [-1.0, 1.0]}


def test_an_unknown_algorithm_fails_before_a_run_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown algorithm"):
        rl_run.main(algorithm="alphazero")
    assert SqliteRunRegistry(tmp_path / "runs.db").list_runs() == []


def test_a_q_learning_run_is_a_leaderboard_entrant_with_a_table_label(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(rl_run, "MONITOR_SEEDS", (20_000, 20_001))
    run_id = rl_run.main(
        algorithm="q_learning", iterations=2, steps_per_iteration=5_000, params={"epsilon_decay_steps": 5_000}
    )
    registry, metrics = SqliteRunRegistry(tmp_path / "runs.db"), FileMetricsStore(tmp_path / "metrics")
    assert registry.get_run(run_id).config["params"] == {"epsilon_decay_steps": 5000.0}
    (entrant,) = evaluate.champion_entrants(registry, metrics, FileArtifactStore(tmp_path / "artifacts"), "snake")
    assert entrant["entrant_id"] == f"run:{run_id}" and entrant["label"].startswith("Q-learning table 2048 states")
    assert entrant["parameters"] == 2048 * 3
    shape = entrant["shape"]
    assert shape["algorithm"] == "Q-learning" and shape["table_states"] == 2048 and 0 < shape["visited_states"] <= 2048
    assert entrant["factory"](0)([0.0] * 11) in (-1, 0, 1)  # plays through the interface's action adapter

    # a run that failed (or was stopped) isn't one: its champion is whatever it had when it stopped
    registry.update_status(run_id, "failed")
    assert evaluate.champion_entrants(registry, metrics, FileArtifactStore(tmp_path / "artifacts"), "snake") == []


def test_snapshot_every_stores_fewer_tables_and_points_between_ones_at_the_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(rl_run, "MONITOR_SEEDS", (20_000,))
    run_id = rl_run.main(algorithm="q_learning", iterations=5, steps_per_iteration=1_000, snapshot_every=3)
    history = FileMetricsStore(tmp_path / "metrics").history(run_id)
    assert [h.champion_ref.rsplit("-gen", 1)[1] for h in history] == ["0", "0", "0", "3", "4"]
    stored = sorted(p.name for p in (tmp_path / "artifacts").rglob(f"{run_id}-gen*"))
    assert stored == [f"{run_id}-gen0", f"{run_id}-gen3", f"{run_id}-gen4"]
