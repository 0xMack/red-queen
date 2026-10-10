import arena.snake
import snake_neuro_run
from telemetry import FileMetricsStore, SqliteRunRegistry


def test_training_run_records_strategy_and_held_out_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(snake_neuro_run, "POPULATION_SIZE", 6)
    monkeypatch.setattr(
        snake_neuro_run,
        "monitor_score",
        lambda interface, policy: arena.snake.monitor_score(interface, policy, (20_000, 20_001)),
    )

    snake_neuro_run.main(seed_strategy="resample:2", held_out_every=2, generations=3)

    run = SqliteRunRegistry(tmp_path / "runs.db").list_runs()[0]
    history = FileMetricsStore(tmp_path / "metrics").history(run.run_id)
    assert run.config["seed_strategy"] == "resample:2" and run.config["training_seeds"] is None
    # every 2nd generation plus the last one
    assert [h.held_out_score is not None for h in history] == [True, False, True]
    assert run.summary["held_out_score"] == history[-1].held_out_score
    assert run.summary["cost"]["episodes"] == 6 * 3 * 2  # genomes x generations x seeds
