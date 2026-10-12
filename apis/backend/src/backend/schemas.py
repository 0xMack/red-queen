"""API-only pydantic models -- there's no `telemetry` equivalent to reuse directly here, unlike the rest
of routers/runs.py (doc 0005's "Reusing existing pydantic models directly" only applies where a
canonical model already exists). The write side (`RunCreate`, `RunPatch`, `PruneRequest`) is what the HTTP sink
(jobcore.sink, docs/design/0018) sends.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from telemetry import GenerationStats, RunStatus


class ControlAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"


class ControlRequest(BaseModel):
    action: ControlAction = Field(..., description="Control action to apply to a running evolve() call.")


class RunSummary(BaseModel):
    """What a list of runs shows for each one -- a few numbers and a short trend, not its whole history. The runs
    page used to download every run's full history (13 MB for 115 runs) to draw 60-point sparklines."""

    run_id: str = Field(..., min_length=1)
    generations: int = Field(..., ge=0, description="Generations (or RL iterations) recorded so far.")
    best_fitness: float | None = Field(
        ..., description="Highest best_fitness over the whole run (training fitness / return); None if none recorded."
    )
    last: GenerationStats | None = Field(
        ..., description="The latest generation's stats (its extras, its timestamp); None if none recorded."
    )
    trend: list[float] = Field(
        ...,
        description="best_fitness over the run, downsampled to at most `trend_points` evenly spaced generations "
        "(first and last always included).",
    )


class RunCreate(BaseModel):
    """A run as a job starts it (the HTTP sink's `create_run`, docs/design/0018)."""

    config: dict[str, Any] = Field(..., description="The run's configuration, recorded as given.")


class RunCreated(BaseModel):
    run_id: str = Field(..., min_length=1)


class RunPatch(BaseModel):
    """Any of a run's mutable parts. `config` is *merged* (top-level keys; None removes one) -- for corrections after
    the fact, like a run recorded before a field existed."""

    status: RunStatus | None = None
    summary: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class PruneRequest(BaseModel):
    protocol: str = Field(..., min_length=1)
    keep: list[str] = Field(..., description="Entrant ids that still qualify; every other record of the protocol goes.")


class PruneResult(BaseModel):
    removed: list[str]
