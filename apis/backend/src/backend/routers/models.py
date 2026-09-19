"""Model packages for client-side inference (docs/design/0009 Decision 5).

In development this backend serves the local model store's files directly; in a deployment the same
tree is uploaded to a bucket/CDN, `REDQUEEN_MODELS_BASE_URL` points clients there, and only the
catalog route stays here. Blobs and manifests are content-addressed, so they're served as immutable:
a browser or CDN never has to revalidate them. Only the catalog changes.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from modelpack import build_package, export_network_json
from modelpack.store import GAME, SHA256

from backend.dependencies import ArtifactStoreDep, LocalModelStoreDep
from backend.settings import models_base_url

IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}

router = APIRouter(prefix="/models", tags=["models"])
catalog_router = APIRouter(prefix="/games", tags=["models"])
export_router = APIRouter(prefix="/runs", tags=["models"])


def _base_url(request: Request) -> str:
    return models_base_url() or str(request.url_for("get_blob", sha256="x")).removesuffix("/blobs/x")


@router.get("/blobs/{sha256}")
def get_blob(sha256: str, store: LocalModelStoreDep) -> FileResponse:
    if not SHA256.match(sha256) or not store.has_blob(sha256):
        raise HTTPException(status_code=404, detail="no such blob")
    return FileResponse(store.blob_path(sha256), media_type="application/octet-stream", headers=IMMUTABLE)


@router.get("/manifests/{package_id}.json")
def get_manifest(package_id: str, store: LocalModelStoreDep) -> FileResponse:
    if not SHA256.match(package_id) or not store.manifest_path(package_id).exists():
        raise HTTPException(status_code=404, detail="no such package")
    return FileResponse(store.manifest_path(package_id), media_type="application/json", headers=IMMUTABLE)


@catalog_router.get("/{game}/models")
def get_catalog(game: str, request: Request, store: LocalModelStoreDep) -> dict[str, Any]:
    """Published packages for `game`, per leaderboard entrant, plus `base_url` -- where `blobs/<sha>`
    and `manifests/<id>.json` are fetched from. Empty (not 404) for a game with nothing published."""
    if not GAME.match(game):
        raise HTTPException(status_code=404, detail="no such game")
    catalog = store.catalog(game)
    return {"game": game, "base_url": _base_url(request), "entries": [e.model_dump(mode="json") for e in catalog.entries]}


@export_router.post("/{run_id}/artifacts/{ref}/package")
def export_champion(run_id: str, ref: str, request: Request, artifacts: ArtifactStoreDep, store: LocalModelStoreDep) -> dict[str, Any]:
    """Package any stored champion on demand -- how the browser watches a champion that was never
    published (a still-training run's latest, a pinned generation). Development/live-training only:
    it costs a little server compute per *new* champion, never per frame, and packages are
    content-addressed, so asking again for the same champion re-uses the stored one. Unlike
    jobs/publish_models.py it measures only numeric parity (on random inputs), not action agreement
    on the protocol's games -- so it's marked `on_demand`, and never becomes a leaderboard entrant."""
    # Champion refs are "<run_id>-gen<N>" (jobs/*_run.py); anything else isn't this run's champion.
    if not ref.startswith(f"{run_id}-"):
        raise HTTPException(status_code=404, detail=f"{ref} isn't a champion of run {run_id}")
    try:
        raw = artifacts.get_program(ref)
    except OSError:
        raise HTTPException(status_code=404, detail=f"no such program artifact: {ref}") from None
    try:
        text = raw.decode("utf-8")
        package = build_package(
            [export_network_json(text, "float64"), export_network_json(text, "float32")],
            label=f"{run_id[:8]} · {ref.removeprefix(run_id).lstrip('-')} (on-demand export)",
            run_id=run_id,
            champion_ref=ref,
        )
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(status_code=422, detail=f"can't export {ref}: {e}") from None
    store.put(package)
    manifest = package.manifest
    return {
        "base_url": _base_url(request),
        "package_id": manifest.package_id,
        "variants": [v.id for v in manifest.variants],
        "on_demand": True,
    }
