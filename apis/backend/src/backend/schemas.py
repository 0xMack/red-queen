"""API-only pydantic models -- there's no `telemetry` equivalent to reuse directly here, unlike the rest
of routers/runs.py (doc 0005's "Reusing existing pydantic models directly" only applies where a
canonical model already exists).
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ControlAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"


class ControlRequest(BaseModel):
    action: ControlAction = Field(..., description="Control action to apply to a running evolve() call.")
