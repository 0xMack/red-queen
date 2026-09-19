"""Leaderboards and representations per game (docs/design/0007).

Read-only: evaluations are produced by jobs/evaluate.py (a separate evaluation job, never training
fitness), and interfaces are defined in games.interfaces. This router just serves both.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from games import interfaces
from telemetry import EvaluationRecord

from backend.dependencies import EvaluationStoreDep

router = APIRouter(prefix="/games", tags=["leaderboards"])


def _mean(record: EvaluationRecord) -> float:
    return float(record.metrics.get("quality", {}).get("mean", float("-inf")))


@router.get("/{game}/leaderboard")
def get_leaderboard(game: str, store: EvaluationStoreDep, protocol: str | None = None) -> list[EvaluationRecord]:
    """Every evaluation of `game` (optionally one protocol), best mean score first. Records from
    different protocols aren't comparable -- a client showing more than one groups by protocol."""
    return sorted(store.list(game, protocol), key=_mean, reverse=True)


@router.get("/{game}/interfaces")
def get_interfaces(game: str) -> list[dict[str, Any]]:
    """The representations (observer + action adapter) a model can be trained under for `game`."""
    found = interfaces.for_game(game)
    if not found:
        raise HTTPException(status_code=404, detail=f"no interfaces registered for game: {game}")
    return [interface.describe() for interface in found]
