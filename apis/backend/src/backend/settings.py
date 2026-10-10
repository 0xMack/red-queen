"""Settings that are this backend's own. Where the data lives is not one of them: that is `telemetry.data_dir()` /
`telemetry.models_dir()`, shared with every job (docs/design/0018), so the two can never disagree.

This is a local, single-user tool (docs/design/0005 "Auth/deployment: out of scope") -- one data directory, no
multi-tenancy or per-request routing.
"""

import os


def models_base_url() -> str | None:
    """Where clients fetch package blobs and manifests from. Unset (development) means this backend's
    own `/models` routes; a deployment points it at the bucket/CDN the store was uploaded to."""
    return os.environ.get("REDQUEEN_MODELS_BASE_URL") or None
