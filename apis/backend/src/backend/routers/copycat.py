"""Copy Cat's held-out images and every entrant's SVG for each (docs/design/0022).

The leaderboard is the generic `/games/copycat/leaderboard`, except that its records leave out each image's SVG -- the
outputs are megabytes, and a leaderboard is read on every page load. They are here instead, one image at a time:

- `GET /games/copycat/targets`: the held-out images of the latest protocol (`arena.copycat.PROTOCOLS`), each as its PNG,
  for picking one. An artwork set that isn't built here (`redqueen dev fetch-sets`) is left out, never downloaded;
- `GET /games/copycat/targets/{set_name}/{seed}`: one image -- its PNG, its clean SVG, and every entrant's SVG and scores
  for it. Seeds repeat across sets, so an image is named by both.

"The latest protocol" is the page's: the last one with records (`latestProtocol` in the frontend), else the newest.
"""

import base64
from typing import Any

from arena import copycat, copycat_artwork
from arena.copycat_sets import group_targets
from fastapi import APIRouter, HTTPException
from jobcore import experiments_dir
from telemetry import EvaluationRecord, EvaluationStore

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


def png_url(target: copycat.Target) -> str:
    return "data:image/png;base64," + base64.b64encode(target.png).decode()


def _source(target: copycat.Target) -> dict[str, Any] | None:
    """Where an artwork image's clean SVG came from, with its credit and licence (`None` for our own images)."""
    if target.icon is None:
        return None
    family = copycat_artwork.family_of(target.set_name, target.icon)
    name = target.icon.removeprefix(family["prefix"])
    return {"url": f"{family['base']}/{name}.svg", "credit": family["credit"], "licence": family["licence"]}


def _describe(target: copycat.Target) -> dict[str, Any]:
    return {
        "id": target.id,
        "set": target.set_name,
        "seed": target.seed,
        "severity": target.severity,
        "levels": target.levels,
        "width": target.width,
        "height": target.height,
        "png": png_url(target),
        "source": _source(target),
    }


def _latest(store: EvaluationStore) -> tuple[str, list[EvaluationRecord]]:
    for protocol in reversed(copycat.PROTOCOLS):
        if records := store.list(copycat.GAME, protocol):
            return protocol, records
    return copycat.PROTOCOL, []


def _targets(sets: tuple[str, ...]) -> list[copycat.Target]:
    targets = []
    for set_name in sets:
        try:
            targets += group_targets(f"{set_name}/{copycat.SPLIT}", experiments_dir() / "copycat-sets", download=False)
        except copycat_artwork.SetNotFetched:
            continue
    return targets


@router.get("/targets")
def get_targets(store: EvaluationStoreDep) -> list[dict[str, Any]]:
    """The held-out images of the latest protocol, set by set in seed order, each as a PNG data URL with its corruption
    levels and, for artwork, where it came from."""
    return [_describe(t) for t in _targets(copycat.PROTOCOLS[_latest(store)[0]])]


@router.get("/targets/{set_name}/{seed}")
def get_target(set_name: str, seed: int, store: EvaluationStoreDep) -> dict[str, Any]:
    """One held-out image: its PNG, the clean SVG it was made from, and per entrant (of the latest protocol's records)
    the SVG it wrote for this image and that SVG's scores."""
    protocol, records = _latest(store)
    sets = (set_name,) if set_name in copycat.PROTOCOLS[protocol] else ()
    target = next((t for t in _targets(sets) if t.seed == seed), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"no held-out Copy Cat image {set_name}/{seed}")
    outputs: dict[str, dict[str, Any]] = {}
    for record in records:
        image = next((i for i in record.metrics.get("copycat", {}).get("images", []) if i["id"] == target.id), None)
        if image is not None and "svg" in image:
            outputs[record.entrant_id] = {"svg": image["svg"], "scores": _scores(image)}
    return {**_describe(target), "clean": target.clean, "protocol": protocol, "outputs": outputs}
