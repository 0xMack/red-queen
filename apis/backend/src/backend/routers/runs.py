"""Run/metrics/artifact endpoints -- doc 0005's "routers/runs.py" group.

Reuses telemetry's own pydantic models (`RunInfo`, `GenerationStats`) directly as response models,
per doc 0005 "Reusing existing pydantic models directly" -- no duplicate API-layer schemas for data
that already has a canonical shape.
"""

import json
import time
from typing import Annotated, Literal

import anyio
from evolve.networks import compiled, network_from_json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse
from telemetry import GenerationStats, MetricsSource, RunInfo, RunRegistry

from backend.dependencies import ArtifactStoreDep, MetricsSourceDep, RunRegistryDep
from backend.schemas import ControlAction, ControlRequest, RunSummary

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def list_runs(registry: RunRegistryDep) -> list[RunInfo]:
    return registry.list_runs()


def _downsample(values: list[float], points: int) -> list[float]:
    """At most `points` evenly spaced values, first and last included."""
    if len(values) <= points:
        return values
    if points == 1:
        return [values[-1]]
    return [values[round(i * (len(values) - 1) / (points - 1))] for i in range(points)]


@router.get("/summaries")
def list_run_summaries(
    registry: RunRegistryDep,
    metrics: MetricsSourceDep,
    trend_points: Annotated[int, Query(ge=1, le=500)] = 60,
) -> list[RunSummary]:
    """One summary per run, in `list_runs()` order: what a runs list needs, in one request."""
    summaries = []
    for run in registry.list_runs():
        history = metrics.history(run.run_id)
        best = [h.best_fitness for h in history]
        summaries.append(
            RunSummary(
                run_id=run.run_id,
                generations=len(history),
                best_fitness=max(best) if best else None,
                last=history[-1] if history else None,
                trend=_downsample(best, trend_points),
            )
        )
    return summaries


@router.get("/{run_id}")
def get_run(run_id: str, registry: RunRegistryDep) -> RunInfo:
    try:
        return registry.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None


@router.get("/{run_id}/metrics/history")
def get_metrics_history(
    run_id: str,
    metrics: MetricsSourceDep,
    since_generation: Annotated[int, Query(ge=0)] = 0,
) -> list[GenerationStats]:
    return metrics.history(run_id, since_generation=since_generation)


async def _stream_generation_stats(metrics: MetricsSource, run_id: str, since_generation: int):
    # subscribe() is a synchronous, never-returning generator that blocks on time.sleep() between
    # polls (see telemetry/metrics.py). Offload each next() call to a worker thread so the blocking
    # sleep doesn't stall the event loop; abandon_on_cancel=True lets a client disconnect cut this
    # short instead of waiting out a full poll interval.
    iterator = metrics.subscribe(run_id, since_generation=since_generation)
    while True:
        stats = await anyio.to_thread.run_sync(next, iterator, abandon_on_cancel=True)
        yield {"event": "generation", "data": stats.model_dump_json()}


@router.get("/{run_id}/metrics/stream")
async def stream_metrics(
    run_id: str,
    metrics: MetricsSourceDep,
    since_generation: Annotated[int, Query(ge=0)] = 0,
) -> EventSourceResponse:
    return EventSourceResponse(_stream_generation_stats(metrics, run_id, since_generation))


@router.post("/{run_id}/control", status_code=202)
def control_run(
    run_id: str,
    body: ControlRequest,
    registry: RunRegistryDep,
    metrics: MetricsSourceDep,
) -> None:
    try:
        registry.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None

    if body.action == ControlAction.PAUSE:
        registry.update_status(run_id, "paused")
    elif body.action == ControlAction.RESUME:
        registry.update_status(run_id, "running")
    elif body.action == ControlAction.STEP:
        _step_one_generation(registry, metrics, run_id)


def _step_one_generation(
    registry: RunRegistry,
    metrics: MetricsSource,
    run_id: str,
    timeout_s: float = 30.0,
    poll_interval: float = 0.1,
) -> None:
    """Resumes a paused run just long enough for exactly one more generation to be recorded, then
    re-pauses it.

    Implemented entirely here rather than as a third RunStatus value -- the training job's control
    callback (jobs/control.py) only ever needs to understand "paused" vs. everything else; "step" is
    this endpoint driving that same two-state mechanism from the outside.
    """
    last_generation = max((s.generation for s in metrics.history(run_id)), default=-1)
    registry.update_status(run_id, "running")
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            if metrics.history(run_id, since_generation=last_generation + 1):
                return
            time.sleep(poll_interval)
        raise HTTPException(
            status_code=504,
            detail=f"timed out waiting for run {run_id} to advance one generation",
        )
    finally:
        registry.update_status(run_id, "paused")


@router.get("/{run_id}/artifacts/{ref}")
def get_artifact(
    run_id: str,  # unused: ArtifactStore isn't run-scoped (flat ref -> bytes); kept in the URL
    # for REST grouping/readability, matching doc 0005's endpoint table.
    ref: str,
    artifacts: ArtifactStoreDep,
    kind: Literal["program", "trace"] = "program",
) -> Response:
    getter = artifacts.get_program if kind == "program" else artifacts.get_trace
    try:
        data = getter(ref)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no such {kind} artifact: {ref}") from None
    return Response(content=data, media_type="application/octet-stream")


@router.get("/{run_id}/artifacts/{ref}/brain")
def get_brain(run_id: str, ref: str, artifacts: ArtifactStoreDep) -> dict:
    """A champion as plain numbers a game core can be built from (`evolve.networks.compiled`): a layered
    network's weights, or a NEAT genome's compiled evaluation plan. What the browser feeds the Rust
    `CheckersStrategy` -- so no client re-implements NEAT or the weight layout."""
    try:
        data = artifacts.get_program(ref)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no such program artifact: {ref}") from None
    try:
        network = network_from_json(data.decode("utf-8"))
        brain = compiled(network)
        if brain["kind"] == "graph":
            # The compiled plan is what the game core evaluates; the genome itself is what a diagram draws (nodes,
            # depths, disabled genes) -- so a NEAT champion carries both.
            brain["genome"] = json.loads(network.to_json())
        return brain
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=f"artifact {ref} isn't a trained network: {e}") from None
