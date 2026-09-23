"""API-only pydantic models -- there's no `telemetry` equivalent to reuse directly here, unlike the rest
of routers/runs.py (doc 0005's "Reusing existing pydantic models directly" only applies where a
canonical model already exists).
"""

from enum import StrEnum

from pydantic import BaseModel, Field
from telemetry import GenerationStats


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
