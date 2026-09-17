"""Run/metrics/artifact endpoints -- doc 0005's "routers/runs.py" group.

Reuses telemetry's own pydantic models (`RunInfo`, `GenerationStats`) directly as response models,
per doc 0005 "Reusing existing pydantic models directly" -- no duplicate API-layer schemas for data
that already has a canonical shape.
"""

from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse
from telemetry import GenerationStats, MetricsSource, RunInfo

from backend.dependencies import ArtifactStoreDep, MetricsSourceDep, RunRegistryDep

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def list_runs(registry: RunRegistryDep) -> list[RunInfo]:
    return registry.list_runs()


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


async def _stream_generation_stats(
    metrics: MetricsSource, run_id: str, since_generation: int
):
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
    return EventSourceResponse(
        _stream_generation_stats(metrics, run_id, since_generation)
    )


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
        raise HTTPException(
            status_code=404, detail=f"no such {kind} artifact: {ref}"
        ) from None
    return Response(content=data, media_type="application/octet-stream")
