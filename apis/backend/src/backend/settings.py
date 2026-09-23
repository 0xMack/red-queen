"""Where to find telemetry data on disk.

This is a local, single-user tool (docs/design/0005 "Auth/deployment: out of scope") -- one run-data
directory, configurable via an env var, no multi-tenancy or per-request routing.
"""

import os
from pathlib import Path


def run_data_dir() -> Path:
    """Directory containing runs.db/metrics/artifacts, as written by jobs/*.py scripts.

    Defaults to jobs/run-data relative to the current working directory, matching
    jobs/run_context.py's RUN_DATA_DIR -- both are meant to be run from the repo root.
    Override with REDQUEEN_RUN_DATA_DIR to point at a different run's data.
    """
    override = os.environ.get("REDQUEEN_RUN_DATA_DIR")
    return Path(override) if override else Path("jobs/run-data")


def models_dir() -> Path:
    """The local model store (docs/design/0009) -- written by jobs/publish_models.py. Override with
    REDQUEEN_MODELS_DIR."""
    override = os.environ.get("REDQUEEN_MODELS_DIR")
    return Path(override) if override else run_data_dir() / "models"


def models_base_url() -> str | None:
    """Where clients fetch package blobs and manifests from. Unset (development) means this backend's
    own `/models` routes; a deployment points it at the bucket/CDN the store was uploaded to."""
    return os.environ.get("REDQUEEN_MODELS_BASE_URL") or None
