"""FastAPI dependency providers.

Telemetry-backed dependencies use the Protocol types (`RunRegistry`/`MetricsSource`/
`ArtifactStore`), not the concrete `Sqlite*`/`File*` classes -- so swapping storage backends later
(docs/design/0002) never touches router code, only the factory functions here. `GameSessionStore`
has no such Protocol -- it's in-memory, single-implementation, app-specific state (game_sessions.py).
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from telemetry import (
    ArtifactStore,
    EvaluationStore,
    FileArtifactStore,
    FileMetricsStore,
    MetricsSource,
    RunRegistry,
    SqliteEvaluationStore,
    SqliteRunRegistry,
)

from backend.game_sessions import GameSessionStore
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


@lru_cache
def _evaluation_store() -> SqliteEvaluationStore:
    return SqliteEvaluationStore(run_data_dir() / "evaluations.db")


@lru_cache
def _game_session_store() -> GameSessionStore:
    return GameSessionStore()


RunRegistryDep = Annotated[RunRegistry, Depends(_run_registry)]
MetricsSourceDep = Annotated[MetricsSource, Depends(_metrics_source)]
ArtifactStoreDep = Annotated[ArtifactStore, Depends(_artifact_store)]
EvaluationStoreDep = Annotated[EvaluationStore, Depends(_evaluation_store)]
GameSessionStoreDep = Annotated[GameSessionStore, Depends(_game_session_store)]
