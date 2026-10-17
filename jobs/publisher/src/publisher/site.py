"""The `site` target: the API's public GET routes as static files, for hosting the site with no server
(docs/design/0019). The rendering is the backend's own (`backend.static_site`): this only says where it goes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobcore import Sink
from jobcore.specs import PublishSpec
from modelpack import LocalModelStore
from telemetry import data_dir

from publisher.registry import target


@target("site")
def publish_site(spec: PublishSpec, params: Any, sink: Sink, store: LocalModelStore) -> dict[str, Any]:
    from backend.static_site import export

    out = Path(params.out) if params.out else data_dir() / "site"
    return export(out, params.models_base_url, include_scale_test=params.include_scale_test)
