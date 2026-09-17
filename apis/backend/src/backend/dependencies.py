"""FastAPI dependency providers over telemetry's storage backends.

Routers depend on the Protocol types (`RunRegistry`/`MetricsSource`/`ArtifactStore`), not the
concrete `Sqlite*`/`File*` classes -- so swapping storage backends later (docs/design/0002) never
touches router code, only the factory functions here.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from telemetry import (
    ArtifactStore,
    FileArtifactStore,
    FileMetricsStore,
    MetricsSource,
    RunRegistry,
    SqliteRunRegistry,
)

from backend.settings import run_data_dir


@lru_cache
def _run_registry() -> SqliteRunRegistry:
    return SqliteRunRegistry(run_data_dir() / "runs.db")


@lru_cache
def _metrics_source() -> FileMetricsStore:
    return FileMetricsStore(run_data_dir() / "metrics")


@lru_cache
def _artifact_store() -> FileArtifactStore:
    return FileArtifactStore(run_data_dir() / "artifacts")


RunRegistryDep = Annotated[RunRegistry, Depends(_run_registry)]
MetricsSourceDep = Annotated[MetricsSource, Depends(_metrics_source)]
ArtifactStoreDep = Annotated[ArtifactStore, Depends(_artifact_store)]
