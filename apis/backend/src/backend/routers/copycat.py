"""Copy Cat's held-out images and every entrant's SVG for each (docs/design/0022).

The leaderboard is the generic `/games/copycat/leaderboard`, except that its records leave out each image's SVG -- the
outputs are megabytes, and a leaderboard is read on every page load. They are here instead, one image at a time:

- `GET /games/copycat/targets`: the held-out images (`arena.copycat.load_set`), each as its PNG, for picking one;
- `GET /games/copycat/targets/{seed}`: one image -- its PNG, its clean SVG, and every entrant's SVG and scores for it.
"""

import base64
from typing import Any

from arena import copycat
from fastapi import APIRouter, HTTPException
from telemetry import EvaluationRecord

from backend.dependencies import EvaluationStoreDep

router = APIRouter(prefix="/games/copycat", tags=["copycat"])


def without_outputs(record: EvaluationRecord) -> EvaluationRecord:
    """The record without its per-image SVGs (what the leaderboard serves)."""
    images = record.metrics.get("copycat", {}).get("images")
    if not images or "svg" not in images[0]:
        return record
    metrics = {**record.metrics, "copycat": {**record.metrics["copycat"], "images": [_scores(i) for i in images]}}
    return record.model_copy(update={"metrics": metrics})


def _scores(image: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in image.items() if k != "svg"}


def _png(target: copycat.Target) -> str:
    return "data:image/png;base64," + base64.b64encode(target.png).decode()


def _describe(target: copycat.Target) -> dict[str, Any]:
    return {
        "id": target.id,
        "seed": target.seed,
        "severity": target.severity,
        "levels": target.levels,
        "width": target.width,
        "height": target.height,
        "png": _png(target),
    }


@router.get("/targets")
def get_targets() -> list[dict[str, Any]]:
    """The held-out images of `copycat.clean.v1`, in seed order, each as a PNG data URL with its corruption levels."""
    return [_describe(t) for t in copycat.load_set()]


@router.get("/targets/{seed}")
def get_target(seed: int, store: EvaluationStoreDep) -> dict[str, Any]:
    """One held-out image: its PNG, the clean SVG it was made from, and per entrant (of the protocol's records) the SVG
    it wrote for this image and that SVG's scores."""
    target = next((t for t in copycat.load_set() if t.seed == seed), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"no held-out Copy Cat image with seed {seed}")
    outputs: dict[str, dict[str, Any]] = {}
    for record in store.list(copycat.GAME, copycat.PROTOCOL):
        image = next((i for i in record.metrics.get("copycat", {}).get("images", []) if i["id"] == target.id), None)
        if image is not None and "svg" in image:
            outputs[record.entrant_id] = {"svg": image["svg"], "scores": _scores(image)}
    return {**_describe(target), "clean": target.clean, "protocol": copycat.PROTOCOL, "outputs": outputs}
