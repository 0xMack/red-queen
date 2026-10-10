"""Where a workload writes: the run sink (docs/design/0018, decision 3).

A sink is telemetry's `Stores`: every store, typed by its Protocol (docs/design/0002). The local sink is the
file/SQLite stores under `telemetry.data_dir()` -- log-then-serve, so a run never depends on the backend being up.
The HTTP sink (stage 5) will be HTTP implementations of the same Protocols, for a worker that can't share the data
directory; nothing that holds a sink changes when it arrives.

Every workload gets its stores here, never from `telemetry.open_stores()` directly, so choosing a sink stays one
decision in one place.
"""

from __future__ import annotations

import os

from telemetry import Stores, open_stores

Sink = Stores


def open_sink(target: str | None = None) -> Sink:
    """The sink named by `target`, else `REDQUEEN_SINK`, else `local`."""
    target = target or os.environ.get("REDQUEEN_SINK") or "local"
    if target == "local":
        return open_stores()
    raise ValueError(
        f"unknown sink {target!r}: only 'local' exists until the backend's write endpoints do (docs/design/0018 stage 5)"
    )
