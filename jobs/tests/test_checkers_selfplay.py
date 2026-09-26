import checkers_selfplay_run
import evaluate_versus
import run_context
from evolve import WeightVector
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry


def test_a_self_play_run_is_recorded_and_joins_the_versus_leaderboard(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(checkers_selfplay_run, "MONITOR_GAMES", 2)

    run_id = checkers_selfplay_run.main(
        iterations=evaluate_versus.MIN_GENERATIONS, games_per_iteration=20, depth=1, held_out_every=5, rng_seed=3
    )

    registry, metrics = SqliteRunRegistry(tmp_path / "runs.db"), FileMetricsStore(tmp_path / "metrics")
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    run = registry.get_run(run_id)
    assert run.status == "completed" and run.config["representation"] == "td_lambda"
    assert run.config["interface"] == evaluate_versus.INTERFACE and run.config["layer_sizes"] == [32, 16, 1]
    history = metrics.history(run_id)
    assert len(history) == evaluate_versus.MIN_GENERATIONS
    assert [h.held_out_score is not None for h in history][:6] == [True, False, False, False, False, True]
    assert history[-1].extras["games"] == 20 * evaluate_versus.MIN_GENERATIONS
    assert -1 <= history[-1].mean_fitness <= 1 and history[-1].extras["td_loss"] >= 0
    champion = WeightVector.from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    assert champion.layer_sizes == (32, 16, 1)
    assert run.summary["cost"]["episodes"] == 20 * evaluate_versus.MIN_GENERATIONS

    (entrant,) = evaluate_versus.champion_entrants(registry, metrics, artifacts)
    assert entrant["label"].startswith("TD(λ) self-play 32 → 16 → 1") and "self-play" in entrant["model"]
