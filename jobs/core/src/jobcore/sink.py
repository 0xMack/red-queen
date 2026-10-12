"""Where a workload writes: the run sink (docs/design/0018, decision 3).

A sink is telemetry's `Stores`: every store, typed by its Protocol (docs/design/0002). The local sink is the
file/SQLite stores under `telemetry.data_dir()` -- log-then-serve, so a run never depends on the backend being up.
The HTTP sink (`jobcore.http`) is HTTP implementations of the same Protocols over the backend's API, for a worker
that can't share the data directory; nothing that holds a sink can tell the two apart.

Every workload gets its stores here, never from `telemetry.open_stores()` directly, so choosing a sink stays one
decision in one place.
"""

from __future__ import annotations

import os

from telemetry import Stores, open_stores

Sink = Stores


def open_sink(target: str | None = None) -> Sink:
    """The sink named by `target`, else `REDQUEEN_SINK`, else `local`: `local`, or a backend's base URL
    (`http://host:8000`) for the HTTP sink."""
    target = target or os.environ.get("REDQUEEN_SINK") or "local"
    if target == "local":
        return open_stores()
    if target.startswith(("http://", "https://")):
        from jobcore.http import http_sink

        return http_sink(target)
    raise ValueError(f"unknown sink {target!r}: `local`, or a backend URL like http://localhost:8000")
