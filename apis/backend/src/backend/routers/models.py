"""Model packages for client-side inference (docs/design/0009 Decision 5).

In development this backend serves the local model store's files directly; in a deployment the same
tree is uploaded to a bucket/CDN, `REDQUEEN_MODELS_BASE_URL` points clients there, and only the
catalog route stays here. Blobs and manifests are content-addressed, so they're served as immutable:
a browser or CDN never has to revalidate them. Only the catalog changes.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from modelpack.store import GAME, SHA256

from backend.dependencies import LocalModelStoreDep
from backend.settings import models_base_url

IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}

router = APIRouter(prefix="/models", tags=["models"])
catalog_router = APIRouter(prefix="/games", tags=["models"])


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
    base_url = models_base_url() or str(request.url_for("get_blob", sha256="x")).removesuffix("/blobs/x")
    return {"game": game, "base_url": base_url, "entries": [e.model_dump(mode="json") for e in catalog.entries]}
