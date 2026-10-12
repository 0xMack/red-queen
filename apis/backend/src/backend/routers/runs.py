"""Run/metrics/artifact endpoints -- doc 0005's "routers/runs.py" group.

The write side (POST /runs, PATCH /runs/{id}, POST /runs/{id}/metrics, PUT /runs/{id}/artifacts/{ref}) is what a job
on the HTTP sink writes through (docs/design/0018) -- a worker that can't share the data directory. A job on the
local sink writes the same stores directly; either way the stores hold the same thing.

Reuses telemetry's own pydantic models (`RunInfo`, `GenerationStats`) directly as response models,
per doc 0005 "Reusing existing pydantic models directly" -- no duplicate API-layer schemas for data
that already has a canonical shape.
"""

import json
import time
from functools import partial
from typing import Annotated, Literal

import anyio
from evolve.networks import compiled, network_from_json
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse
from telemetry import GenerationStats, MetricsSource, RunInfo, RunRegistry

from backend.dependencies import ArtifactStoreDep, MetricsSourceDep, MetricsStoreDep, RunRegistryDep
from backend.schemas import ControlAction, ControlRequest, RunCreate, RunCreated, RunPatch, RunSummary

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


@router.post("", status_code=201)
def create_run(body: RunCreate, registry: RunRegistryDep) -> RunCreated:
    return RunCreated(run_id=registry.create_run(config=body.config))


@router.patch("/{run_id}")
def patch_run(run_id: str, body: RunPatch, registry: RunRegistryDep) -> RunInfo:
    try:
        registry.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None
    if body.config is not None:
        registry.update_config(run_id, body.config)
    if body.summary is not None:
        registry.set_summary(run_id, body.summary)
    if body.status is not None:
        registry.update_status(run_id, body.status)
    return registry.get_run(run_id)


@router.post("/{run_id}/metrics", status_code=204)
def record_generations(run_id: str, body: list[GenerationStats], metrics: MetricsStoreDep) -> None:
    """Appends generations in order. Each must belong to this run: a mismatched id is a client bug, not data."""
    wrong = [s.generation for s in body if s.run_id != run_id]
    if wrong:
        raise HTTPException(status_code=422, detail=f"generations {wrong} belong to another run")
    for stats in body:
        metrics.record_generation(stats)


@router.put("/{run_id}/artifacts/{ref}", status_code=204)
async def put_artifact(
    run_id: str,  # unused, as on GET: the artifact store is flat (ref -> bytes)
    ref: str,
    request: Request,
    artifacts: ArtifactStoreDep,
    kind: Literal["program", "trace"] = "program",
) -> None:
    data = await request.body()
    putter = artifacts.put_program if kind == "program" else artifacts.put_trace
    await anyio.to_thread.run_sync(putter, ref, data)


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


STREAM_POLL_SECONDS = 0.5


async def _stream_generation_stats(
    metrics: MetricsSource, run_id: str, since_generation: int, poll_seconds: float = STREAM_POLL_SECONDS
):
    # A live tail: poll history() -- incremental, it parses only what was appended -- in a worker thread (file I/O
    # off the event loop), and wait between polls with anyio.sleep, where a client disconnect or a server shutdown
    # cancels it. Not subscribe(): its next() blocks until a generation arrives, which for a finished run is never,
    # so each abandoned call parked a (non-daemon) anyio worker thread forever -- and a `--reload` then waited on
    # it and never restarted the server.
    next_generation = since_generation
    while True:
        stats = await anyio.to_thread.run_sync(partial(metrics.history, run_id, since_generation=next_generation))
        for s in stats:
            if s.generation >= next_generation:
                next_generation = s.generation + 1
                yield {"event": "generation", "data": s.model_dump_json()}
        await anyio.sleep(poll_seconds)


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
    callback (jobcore.control) only ever needs to understand "paused" vs. everything else; "step" is
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
