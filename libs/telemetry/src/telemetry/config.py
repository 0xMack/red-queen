"""Where the local data lives, and the one place that opens its stores (docs/design/0018).

Jobs, the backend and tests all used to work this out for themselves -- with different defaults (relative to the
current directory in one place, to a script's own file in another), so everything had to be launched from the repo
root. Now there is one rule: `REDQUEEN_DATA_DIR` if set, else `<workspace root>/data`.

Read at call time, never at import, so a test (or a smoke run) points everything at a scratch directory with
`monkeypatch.setenv("REDQUEEN_DATA_DIR", ...)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from telemetry.artifacts import ArtifactStore, FileArtifactStore
from telemetry.evaluations import EvaluationStore, SqliteEvaluationStore
from telemetry.metrics import FileMetricsStore, MetricsStore
from telemetry.registry import RunRegistry, SqliteRunRegistry

# REDQUEEN_RUN_DATA_DIR is the name this had before 0018; still honoured so existing shells and scripts keep working.
DATA_DIR_VARS = ("REDQUEEN_DATA_DIR", "REDQUEEN_RUN_DATA_DIR")


def workspace_root(start: Path | None = None) -> Path:
    """The nearest directory at or above `start` (default: the current directory) whose pyproject.toml declares the
    uv workspace. Raises if there is none: a worker outside the repo must be told where its data is."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
            return directory
    raise RuntimeError(f"not inside the red-queen workspace ({here}); set REDQUEEN_DATA_DIR")


def data_dir() -> Path:
    """runs.db, evaluations.db, metrics/, artifacts/, models/ and job outputs live here."""
    for var in DATA_DIR_VARS:
        if value := os.environ.get(var):
            return Path(value)
    return workspace_root() / "data"


def models_dir() -> Path:
    """The local model store (docs/design/0009). `REDQUEEN_MODELS_DIR` overrides it, else `<data dir>/models`."""
    override = os.environ.get("REDQUEEN_MODELS_DIR")
    return Path(override) if override else data_dir() / "models"


@dataclass(frozen=True)
class Stores:
    """Every store, typed by its Protocol (docs/design/0002): `open_stores()` fills it with the local file/SQLite
    implementations, and anything else implementing the Protocols (an HTTP client of the backend, docs/design/0018)
    can stand in without its callers changing."""

    registry: RunRegistry
    metrics: MetricsStore
    artifacts: ArtifactStore
    evaluations: EvaluationStore


def open_stores(root: Path | None = None) -> Stores:
    """Every telemetry store under `root` (default: `data_dir()`)."""
    root = data_dir() if root is None else root
    return Stores(
        registry=SqliteRunRegistry(root / "runs.db"),
        metrics=FileMetricsStore(root / "metrics"),
        artifacts=FileArtifactStore(root / "artifacts"),
        evaluations=SqliteEvaluationStore(root / "evaluations.db"),
    )
