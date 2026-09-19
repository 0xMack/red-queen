"""Where packages live (docs/design/0009 Decision 5). Protocol-first like `telemetry`'s stores: the host
(Cloudflare R2, the Hugging Face Hub, ...) is undecided, and a client only ever sees URLs, so choosing
one is a new `ModelStore` implementation, not a change anywhere else.

Layout, identical for every backend so URLs are just `<base>/<key>`:

    blobs/<sha256>                  immutable: graphs, weight shards, parity fixtures
    manifests/<package_id>.json     immutable: a sealed manifest
    catalog/<game>.json             mutable: which packages are published for a game, per entrant

Only the catalog ever changes, so everything else can be served `Cache-Control: immutable`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from modelpack.manifest import ModelManifest
from modelpack.packaging import Package, sha256

SHA256 = re.compile(r"^[0-9a-f]{64}$")
GAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CatalogEntry(BaseModel):
    """One published model for a game: the leaderboard entrant it plays as, and its package."""

    entrant_id: str
    package_id: str
    label: str
    interface: str | None
    run_id: str | None = None
    champion_ref: str | None = None
    variants: list[str]
    download_bytes: dict[str, int] = Field(default_factory=dict, description="variant id -> bytes")


class Catalog(BaseModel):
    game: str
    entries: list[CatalogEntry] = Field(default_factory=list)

    def upsert(self, entry: CatalogEntry) -> None:
        self.entries = [e for e in self.entries if e.entrant_id != entry.entrant_id] + [entry]
        self.entries.sort(key=lambda e: e.entrant_id)

    def find(self, entrant_id: str) -> CatalogEntry | None:
        return next((e for e in self.entries if e.entrant_id == entrant_id), None)


class ModelStore(Protocol):
    def put(self, package: Package) -> None: ...
    def has_blob(self, sha: str) -> bool: ...
    def read_blob(self, sha: str) -> bytes: ...
    def manifest(self, package_id: str) -> ModelManifest: ...
    def catalog(self, game: str) -> Catalog: ...
    def put_catalog(self, catalog: Catalog) -> None: ...


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class LocalModelStore:
    """A directory in the layout above. `apis/backend` serves it as static files in development; a
    real deployment uploads the same tree to a bucket/CDN."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def blob_path(self, sha: str) -> Path:
        if not SHA256.match(sha):
            raise ValueError(f"not a sha256: {sha!r}")
        return self.root / "blobs" / sha

    def manifest_path(self, package_id: str) -> Path:
        if not SHA256.match(package_id):
            raise ValueError(f"not a package id: {package_id!r}")
        return self.root / "manifests" / f"{package_id}.json"

    def catalog_path(self, game: str) -> Path:
        if not GAME.match(game):
            raise ValueError(f"not a game name: {game!r}")
        return self.root / "catalog" / f"{game}.json"

    def put(self, package: Package) -> None:
        for blob in package.manifest.blobs():
            data = package.blobs[blob.sha256]
            if sha256(data) != blob.sha256:
                raise ValueError(f"blob content doesn't match its name {blob.sha256}")
            if not self.has_blob(blob.sha256):
                _atomic_write(self.blob_path(blob.sha256), data)
        path = self.manifest_path(package.manifest.package_id)
        if not path.exists():
            _atomic_write(path, package.manifest.model_dump_json(indent=2).encode())

    def has_blob(self, sha: str) -> bool:
        return self.blob_path(sha).exists()

    def read_blob(self, sha: str) -> bytes:
        return self.blob_path(sha).read_bytes()

    def manifest(self, package_id: str) -> ModelManifest:
        return ModelManifest.model_validate_json(self.manifest_path(package_id).read_text(encoding="utf-8"))

    def catalog(self, game: str) -> Catalog:
        path = self.catalog_path(game)
        if not path.exists():
            return Catalog(game=game)
        return Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def put_catalog(self, catalog: Catalog) -> None:
        _atomic_write(self.catalog_path(catalog.game), catalog.model_dump_json(indent=2).encode())
