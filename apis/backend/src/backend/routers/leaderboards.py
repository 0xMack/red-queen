"""Leaderboards and representations per game (docs/design/0007).

Evaluations are produced by the evaluator workload (a separate evaluation job under a versioned protocol, never
training fitness) -- written to the store directly, or through PUT /games/{game}/evaluations and .../prune from a
worker on the HTTP sink. Interfaces are defined in games.interfaces. This router serves both.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from games import interfaces
from telemetry import EvaluationRecord

from backend.dependencies import EvaluationStoreDep
from backend.routers.copycat import without_outputs
from backend.schemas import PruneRequest, PruneResult

router = APIRouter(prefix="/games", tags=["leaderboards"])


def _mean(record: EvaluationRecord) -> float:
    return float(record.metrics.get("quality", {}).get("mean", float("-inf")))


@router.get("/{game}/leaderboard")
def get_leaderboard(game: str, store: EvaluationStoreDep, protocol: str | None = None) -> list[EvaluationRecord]:
    """Every evaluation of `game` (optionally one protocol), best mean score first. Records from
    different protocols aren't comparable -- a client showing more than one groups by protocol. Copy Cat's per-image
    SVGs are left out (they are `/games/copycat/targets/{seed}`'s)."""
    return sorted((without_outputs(r) for r in store.list(game, protocol)), key=_mean, reverse=True)


@router.get("/{game}/interfaces")
def get_interfaces(game: str) -> list[dict[str, Any]]:
    """The representations (observer + action adapter) a model can be trained under for `game`."""
    found = interfaces.for_game(game)
    if not found:
        raise HTTPException(status_code=404, detail=f"no interfaces registered for game: {game}")
    return [interface.describe() for interface in found]


@router.put("/{game}/evaluations", status_code=204)
def put_evaluation(game: str, record: EvaluationRecord, store: EvaluationStoreDep) -> None:
    """Stores (replaces) one entrant's record under its protocol -- the evaluator on the HTTP sink writes this way."""
    if record.game != game:
        raise HTTPException(status_code=422, detail=f"a {record.game} record, sent to {game}")
    store.put(record)


@router.post("/{game}/evaluations/prune")
def prune_evaluations(game: str, body: PruneRequest, store: EvaluationStoreDep) -> PruneResult:
    """Drops every record of `protocol` whose entrant isn't in `keep` -- entrants that no longer qualify."""
    return PruneResult(removed=store.prune(game, body.protocol, set(body.keep)))
