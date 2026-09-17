"""API-only pydantic models -- there's no `games` or `telemetry` equivalent to reuse directly here,
unlike routers/runs.py (doc 0005's "Reusing existing pydantic models directly" only applies where a
canonical model already exists).
"""

from typing import Any

from pydantic import BaseModel, Field


class GameSessionCreate(BaseModel):
    game: str = Field(
        ..., min_length=1, description="Registered game name, e.g. 'snake'."
    )
    seed: int | None = Field(
        default=None, description="Fixed seed for reproducibility; random if omitted."
    )


class GameSessionState(BaseModel):
    session_id: str = Field(..., min_length=1)
    game: str = Field(..., min_length=1)
    render_state: dict[str, Any] = Field(
        ...,
        description="JSON-safe render_state() output (see game_sessions.json_safe_render_state).",
    )
    reward: float = Field(
        ...,
        description="Reward from the action that produced this state (0.0 for the initial reset state).",
    )
    done: bool = Field(..., description="Whether the episode has ended.")
    step: int = Field(
        ..., ge=0, description="Number of actions applied so far in this session."
    )


class ActionRequest(BaseModel):
    action: float = Field(
        ..., description="Action value; shape/range depends on the game's action space."
    )


class TrajectoryArtifact(BaseModel):
    run_id: str | None = Field(
        default=None,
        description="Originating run, if this came from training rather than a live/human session.",
    )
    game: str = Field(..., min_length=1)
    seed: int = Field(...)
    states: list[dict[str, Any]] = Field(
        ...,
        description="JSON-safe render_state() output at every step, including the initial reset.",
    )
    actions: list[float] = Field(...)
    rewards: list[float] = Field(...)
