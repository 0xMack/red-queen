import itertools
import time

from telemetry import (
    FileArtifactStore,
    FileMetricsStore,
    GenerationStats,
    SqliteRunRegistry,
)


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


def test_subscribe_yields_generations_appended_after_it_started(tmp_path):
    store = FileMetricsStore(tmp_path / "metrics", poll_interval=0.01)
    store.record_generation(make_stats("run-1", 0))
    stream = store.subscribe("run-1")
    assert next(stream).generation == 0

    store.record_generation(make_stats("run-1", 1))
    store.record_generation(make_stats("run-1", 2))
    assert [next(stream).generation for _ in range(2)] == [1, 2]


def test_a_line_caught_mid_write_is_read_once_it_is_complete(tmp_path):
    store = FileMetricsStore(tmp_path / "metrics", poll_interval=0.01)
    store.record_generation(make_stats("run-1", 0))
    path = tmp_path / "metrics" / "run-1.jsonl"
    line = make_stats("run-1", 1).model_dump_json()
    with path.open("a", encoding="utf-8") as f:
        f.write(line[:20])  # a job's write, caught half done

    assert [s.generation for s in store.history("run-1")] == [0]
    stream = store.subscribe("run-1")
    assert next(stream).generation == 0

    with path.open("a", encoding="utf-8") as f:
        f.write(line[20:] + "\n")
    assert next(stream).generation == 1
    assert [s.generation for s in store.history("run-1")] == [0, 1]


def test_history_reads_only_what_was_appended_and_starts_over_if_the_file_shrinks(tmp_path, monkeypatch):
    store = FileMetricsStore(tmp_path / "metrics")
    store.record_generation(make_stats("run-1", 0))
    assert [s.generation for s in store.history("run-1")] == [0]

    reads = []
    original = store._read_from
    monkeypatch.setattr(store, "_read_from", lambda path, offset=0: reads.append(offset) or original(path, offset))
    assert [s.generation for s in store.history("run-1")] == [0]
    assert reads == []  # unchanged file: served from what was already parsed

    store.record_generation(make_stats("run-1", 1))
    assert [s.generation for s in store.history("run-1")] == [0, 1]
    assert len(reads) == 1 and reads[0] > 0  # only the appended line was read

    path = tmp_path / "metrics" / "run-1.jsonl"
    path.write_text(make_stats("run-1", 7).model_dump_json() + "\n", encoding="utf-8")  # rewritten, shorter
    assert [s.generation for s in store.history("run-1")] == [7]
    assert store.history("missing") == []


def test_artifact_store_round_trip(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    store.put_program("ref-1", b"program-bytes")
    store.put_trace("ref-1", b"trace-bytes")

    assert store.get_program("ref-1") == b"program-bytes"
    assert store.get_trace("ref-1") == b"trace-bytes"


# --- evaluations (docs/design/0007) -----------------------------------------------------------------

from telemetry import EvaluationRecord, SqliteEvaluationStore


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


def test_held_out_score_is_optional_and_round_trips(tmp_path):
    from telemetry import FileMetricsStore, GenerationStats

    store = FileMetricsStore(tmp_path / "metrics")
    base = {
        "run_id": "r",
        "island_id": None,
        "timestamp": 1.0,
        "best_fitness": 1.0,
        "mean_fitness": 0.5,
        "worst_fitness": 0.0,
        "diversity": 0.1,
    }
    store.record_generation(GenerationStats(generation=0, champion_ref="r-gen0", **base))
    store.record_generation(GenerationStats(generation=1, champion_ref="r-gen1", held_out_score=12.5, **base))

    history = store.history("r")

    assert [h.held_out_score for h in history] == [None, 12.5]
