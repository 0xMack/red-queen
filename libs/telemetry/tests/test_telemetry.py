import itertools
import time

from telemetry import FileArtifactStore, FileMetricsStore, GenerationStats, SqliteRunRegistry


def make_stats(run_id: str, generation: int) -> GenerationStats:
    return GenerationStats(
        run_id=run_id,
        island_id=None,
        generation=generation,
        timestamp=time.time(),
        best_fitness=1.0 - generation * 0.1,
        mean_fitness=0.5,
        worst_fitness=0.1,
        diversity=0.3,
        champion_ref=f"{run_id}-gen{generation}",
    )


def test_run_registry(tmp_path):
    registry = SqliteRunRegistry(tmp_path / "runs.db")
    run_id = registry.create_run(config={"population_size": 32})

    run = registry.get_run(run_id)
    assert run.status == "running"
    assert run.config == {"population_size": 32}
    assert run.summary is None

    registry.set_summary(run_id, {"best_fitness": 0.9})
    registry.update_status(run_id, "completed")

    run = registry.get_run(run_id)
    assert run.status == "completed"
    assert run.summary == {"best_fitness": 0.9}

    assert [r.run_id for r in registry.list_runs()] == [run_id]


def test_metrics_history_and_subscribe_backfill(tmp_path):
    store = FileMetricsStore(tmp_path / "metrics", poll_interval=0.01)
    run_id = "run-1"

    for gen in range(3):
        store.record_generation(make_stats(run_id, gen))

    history = store.history(run_id)
    assert [s.generation for s in history] == [0, 1, 2]

    # subscribe() backfills existing generations first, then would keep polling for more —
    # take exactly the 3 we wrote so the test doesn't hang on the live tail.
    seen = list(itertools.islice(store.subscribe(run_id), 3))
    assert [s.generation for s in seen] == [0, 1, 2]

    # since_generation filters out earlier generations from both history() and subscribe()
    assert [s.generation for s in store.history(run_id, since_generation=2)] == [2]


def test_artifact_store_round_trip(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    store.put_program("ref-1", b"program-bytes")
    store.put_trace("ref-1", b"trace-bytes")

    assert store.get_program("ref-1") == b"program-bytes"
    assert store.get_trace("ref-1") == b"trace-bytes"
