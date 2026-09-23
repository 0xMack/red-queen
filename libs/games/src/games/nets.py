"""Trained networks the game core runs itself (`rust/core/src/nets.rs`, docs/design/0009).

A network arrives as plain numbers -- `evolve.networks.compiled()`'s `{"kind": "layered", "weights": [...],
"layer_sizes": [...]}` or `{"kind": "graph", "encoding": [...]}` -- so this package still doesn't depend on
`evolve`. The core computes exactly what the Python forward passes do, bit for bit, which is what lets a
training job hand a whole episode to it (`Snake.play`) without changing a single result.
"""

from __future__ import annotations

from typing import Any

from games import _native

Policy = _native.Policy


def native_policy(compiled: dict[str, Any]) -> Policy:
    """A network the core can run, from `evolve.networks.compiled(network)`."""
    kind = compiled.get("kind")
    if kind == "layered":
        return Policy.layered(list(compiled["weights"]), list(compiled["layer_sizes"]))
    if kind == "graph":
        return Policy.graph(list(compiled["encoding"]))
    raise ValueError(f"unknown network kind: {kind!r} (expected 'layered' or 'graph')")
