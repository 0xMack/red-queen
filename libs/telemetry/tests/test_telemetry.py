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


# --- evaluations (docs/design/0007) -----------------------------------------------------------------

from telemetry import EvaluationRecord, SqliteEvaluationStore  # noqa: E402


def _record(entrant_id: str, protocol: str = "snake.score.v1", mean: float = 1.0, game: str = "snake"):
    return EvaluationRecord(
        game=game,
        protocol=protocol,
        entrant_id=entrant_id,
        entrant_kind="baseline",
        label=entrant_id,
        interface="snake/features.v1+relative3.v1",
        created_at=1.0,
        metrics={"quality": {"mean": mean}},
    )


def test_evaluation_store_replaces_same_protocol_and_entrant(tmp_path):
    store = SqliteEvaluationStore(tmp_path / "evaluations.db")
    store.put(_record("baseline:greedy", mean=1.0))
    store.put(_record("baseline:greedy", mean=2.0))

    records = store.list("snake")

    assert len(records) == 1
    assert records[0].metrics["quality"]["mean"] == 2.0


def test_evaluation_store_filters_by_game_and_protocol(tmp_path):
    store = SqliteEvaluationStore(tmp_path / "evaluations.db")
    store.put(_record("baseline:greedy"))
    store.put(_record("baseline:greedy", protocol="snake.score.v2"))
    store.put(_record("baseline:random", game="checkers", protocol="checkers.rating.v1"))

    assert len(store.list("snake")) == 2
    assert [r.protocol for r in store.list("snake", protocol="snake.score.v2")] == ["snake.score.v2"]
    assert [r.entrant_id for r in store.list("checkers")] == ["baseline:random"]
