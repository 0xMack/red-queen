import threading
import time

import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from telemetry import (
    FileArtifactStore,
    FileMetricsStore,
    GenerationStats,
    SqliteRunRegistry,
)

from backend.dependencies import _artifact_store, _metrics_source, _run_registry
from backend.main import app
from backend.routers.runs import _step_one_generation, _stream_generation_stats


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


@pytest.fixture
def backend(tmp_path):
    registry = SqliteRunRegistry(tmp_path / "runs.db")
    metrics = FileMetricsStore(tmp_path / "metrics", poll_interval=0.01)
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    app.dependency_overrides[_run_registry] = lambda: registry
    app.dependency_overrides[_metrics_source] = lambda: metrics
    app.dependency_overrides[_artifact_store] = lambda: artifacts
    try:
        yield TestClient(app), registry, metrics, artifacts
    finally:
        app.dependency_overrides.clear()


def test_list_and_get_run(backend):
    client, registry, _, _ = backend
    run_id = registry.create_run(config={"population_size": 32})

    resp = client.get("/runs")
    assert resp.status_code == 200
    assert [r["run_id"] for r in resp.json()] == [run_id]

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_summaries_give_each_run_its_counts_best_last_and_a_short_trend(backend):
    client, registry, metrics, _ = backend
    long_run = registry.create_run(config={})
    empty_run = registry.create_run(config={})
    for gen in range(200):
        stats = make_stats(long_run, gen)
        stats.best_fitness = float(gen % 50)  # peaks at 49, ends at 49
        metrics.record_generation(stats)

    resp = client.get("/runs/summaries", params={"trend_points": 10})
    assert resp.status_code == 200
    by_id = {s["run_id"]: s for s in resp.json()}
    assert list(by_id) == [r.run_id for r in registry.list_runs()]
    summary = by_id[long_run]
    assert summary["generations"] == 200 and summary["best_fitness"] == 49.0
    assert summary["last"]["generation"] == 199
    assert len(summary["trend"]) == 10 and summary["trend"][0] == 0.0 and summary["trend"][-1] == 49.0
    assert by_id[empty_run] == {"run_id": empty_run, "generations": 0, "best_fitness": None, "last": None, "trend": []}

    assert client.get("/runs/summaries", params={"trend_points": 0}).status_code == 422


def test_summaries_is_not_mistaken_for_a_run_id(backend):
    client, _, _, _ = backend
    assert client.get("/runs/summaries").json() == []  # routed to the list, not to get_run("summaries")


def test_get_run_404(backend):
    client, _, _, _ = backend
    resp = client.get("/runs/does-not-exist")
    assert resp.status_code == 404


def test_metrics_history(backend):
    client, registry, metrics, _ = backend
    run_id = registry.create_run(config={})
    for gen in range(3):
        metrics.record_generation(make_stats(run_id, gen))

    resp = client.get(f"/runs/{run_id}/metrics/history")
    assert [g["generation"] for g in resp.json()] == [0, 1, 2]

    resp = client.get(f"/runs/{run_id}/metrics/history?since_generation=2")
    assert [g["generation"] for g in resp.json()] == [2]


def test_metrics_stream_route_is_registered():
    # Not exercised via a live TestClient request: the stream never terminates on its own (by
    # design -- it's a live tail), and Starlette's TestClient doesn't reliably simulate a mid-stream
    # client disconnect, so the generator's cancellation path never fires and a real request hangs.
    # The generator behind it is covered directly below instead; this just checks the route exists with the
    # right path/method.
    assert "get" in app.openapi()["paths"]["/runs/{run_id}/metrics/stream"]


def test_stream_generation_stats_backfills_then_cancels_cleanly(tmp_path):
    async def scenario():
        metrics = FileMetricsStore(tmp_path / "metrics", poll_interval=0.01)
        run_id = "run-1"
        for gen in range(3):
            metrics.record_generation(make_stats(run_id, gen))

        gen_iter = _stream_generation_stats(metrics, run_id, since_generation=0, poll_seconds=0.01)
        seen = []
        async for event in gen_iter:
            seen.append(GenerationStats.model_validate_json(event["data"]))
            if len(seen) == 3:
                break
        metrics.record_generation(make_stats(run_id, 3))  # arrives while the stream is live
        seen.append(GenerationStats.model_validate_json((await gen_iter.__anext__())["data"]))
        # The generator is now polling past the last recorded generation (there is no 5th).
        # aclose() must interrupt that cleanly rather than hang the test.
        with anyio.fail_after(2):
            await gen_iter.aclose()
        return seen

    seen = anyio.run(scenario)
    assert [s.generation for s in seen] == [0, 1, 2, 3]


def test_stream_never_parks_a_thread_on_subscribe(tmp_path):
    # A finished run's subscribe().__next__() never returns, and an abandoned call held a non-daemon worker
    # thread forever -- which stopped a `--reload` from ever starting the new server. The stream polls instead.
    class HistoryOnly(FileMetricsStore):
        def subscribe(self, run_id, since_generation=0):
            raise AssertionError("the stream must not use subscribe()")

    async def scenario():
        metrics = HistoryOnly(tmp_path / "metrics")
        metrics.record_generation(make_stats("run-1", 0))
        gen_iter = _stream_generation_stats(metrics, "run-1", since_generation=0, poll_seconds=0.01)
        first = await gen_iter.__anext__()
        with anyio.fail_after(2):
            await gen_iter.aclose()
        return first

    assert GenerationStats.model_validate_json(anyio.run(scenario)["data"]).generation == 0


def test_control_pause_and_resume(backend):
    client, registry, _, _ = backend
    run_id = registry.create_run(config={})

    resp = client.post(f"/runs/{run_id}/control", json={"action": "pause"})
    assert resp.status_code == 202
    assert registry.get_run(run_id).status == "paused"

    resp = client.post(f"/runs/{run_id}/control", json={"action": "resume"})
    assert resp.status_code == 202
    assert registry.get_run(run_id).status == "running"


def test_control_unknown_run_404(backend):
    client, _, _, _ = backend
    resp = client.post("/runs/no-such-run/control", json={"action": "pause"})
    assert resp.status_code == 404


def test_control_step_waits_for_one_more_generation_then_repauses(backend):
    client, registry, metrics, _ = backend
    run_id = registry.create_run(config={})
    metrics.record_generation(make_stats(run_id, 0))
    registry.update_status(run_id, "paused")

    def record_next_generation_shortly():
        time.sleep(0.1)
        metrics.record_generation(make_stats(run_id, 1))

    threading.Thread(target=record_next_generation_shortly, daemon=True).start()

    resp = client.post(f"/runs/{run_id}/control", json={"action": "step"})
    assert resp.status_code == 202
    assert registry.get_run(run_id).status == "paused"
    assert [s.generation for s in metrics.history(run_id)] == [0, 1]


def test_control_step_times_out_if_nothing_advances(backend):
    _, registry, metrics, _ = backend
    run_id = registry.create_run(config={})
    registry.update_status(run_id, "paused")

    with pytest.raises(HTTPException) as exc_info:
        _step_one_generation(registry, metrics, run_id, timeout_s=0.2, poll_interval=0.02)
    assert exc_info.value.status_code == 504
    # Re-paused even after a timeout -- a stalled step must not leave the run stuck "running".
    assert registry.get_run(run_id).status == "paused"


def test_get_artifact(backend):
    client, registry, _, artifacts = backend
    run_id = registry.create_run(config={})
    artifacts.put_program("ref-1", b"program-bytes")
    artifacts.put_trace("ref-1", b"trace-bytes")

    resp = client.get(f"/runs/{run_id}/artifacts/ref-1")
    assert resp.content == b"program-bytes"

    resp = client.get(f"/runs/{run_id}/artifacts/ref-1?kind=trace")
    assert resp.content == b"trace-bytes"

    resp = client.get(f"/runs/{run_id}/artifacts/no-such-ref")
    assert resp.status_code == 404


def test_get_brain_serves_a_champion_as_plain_numbers_for_the_game_core(backend):
    from evolve import InnovationTracker, WeightVector, initial_genome

    client, registry, _, artifacts = backend
    run_id = registry.create_run(config={})
    layered = WeightVector(weights=(0.5,) * 33, layer_sizes=(32, 1))
    artifacts.put_program("layered", layered.to_json().encode())
    genome = initial_genome(32, 1, InnovationTracker(first_hidden_id=34), __import__("random").Random(0))
    artifacts.put_program("graph", genome.to_json().encode())
    artifacts.put_program("junk", b'{"type": "mystery"}')

    body = client.get(f"/runs/{run_id}/artifacts/layered/brain").json()
    assert body == {"kind": "layered", "weights": [0.5] * 33, "layer_sizes": [32, 1]}

    body = client.get(f"/runs/{run_id}/artifacts/graph/brain").json()
    assert body["kind"] == "graph" and body["encoding"] == genome.graph_encoding()
    assert body["genome"]["type"] == "neat" and len(body["genome"]["connections"]) == len(
        genome.connections
    )  # for the diagram

    assert client.get(f"/runs/{run_id}/artifacts/nope/brain").status_code == 404
    assert client.get(f"/runs/{run_id}/artifacts/junk/brain").status_code == 422
