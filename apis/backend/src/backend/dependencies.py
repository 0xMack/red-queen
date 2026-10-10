"""FastAPI dependency providers.

Telemetry-backed dependencies use the Protocol types (`RunRegistry`/`MetricsSource`/
`ArtifactStore`), not the concrete `Sqlite*`/`File*` classes -- so swapping storage backends later
(docs/design/0002) never touches router code, only the factory functions here.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from modelpack import LocalModelStore
from telemetry import (
    ArtifactStore,
    EvaluationStore,
    MetricsSource,
    RunRegistry,
    Stores,
    models_dir,
    open_stores,
)


@lru_cache
def _stores() -> Stores:
    return open_stores()


# One provider per store, so a test overrides just the store it fakes.
def _run_registry() -> RunRegistry:
    return _stores().registry


def _metrics_source() -> MetricsSource:
    return _stores().metrics


def _artifact_store() -> ArtifactStore:
    return _stores().artifacts


def _evaluation_store() -> EvaluationStore:
    return _stores().evaluations


@lru_cache
def _model_store() -> LocalModelStore:
    return LocalModelStore(models_dir())


RunRegistryDep = Annotated[RunRegistry, Depends(_run_registry)]
MetricsSourceDep = Annotated[MetricsSource, Depends(_metrics_source)]
ArtifactStoreDep = Annotated[ArtifactStore, Depends(_artifact_store)]
EvaluationStoreDep = Annotated[EvaluationStore, Depends(_evaluation_store)]
# The concrete local store, not the `ModelStore` protocol: serving its files from disk is exactly what
# a remote store (R2, the HF Hub) would *not* need this backend for -- see routers/models.py.
LocalModelStoreDep = Annotated[LocalModelStore, Depends(_model_store)]
