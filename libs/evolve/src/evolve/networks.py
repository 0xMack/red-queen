"""One loader for every trained-network wire format.

`WeightVector.to_json()` (fixed topology) and `NeatGenome.to_json()` (evolved topology) both end up
as opaque artifacts in `telemetry.ArtifactStore`; whoever reads one back -- `jobs/evaluate.py`,
`jobs/publish_models.py`, `apis/backend`'s on-demand export -- only needs `forward(observation) -> outputs`
(or `compiled()`), so this picks the right class from the payload. NEAT's JSON carries `"type": "neat"`; a WeightVector's never had a
type field (and old artifacts must keep loading), so "no type" means WeightVector.
"""

from __future__ import annotations

import json

from evolve.neat import NeatGenome
from evolve.neuro import WeightVector

Network = WeightVector | NeatGenome


def network_from_json(text: str) -> Network:
    kind = json.loads(text).get("type", "weight_vector")
    if kind == "neat":
        return NeatGenome.from_json(text)
    if kind == "weight_vector":
        return WeightVector.from_json(text)
    raise ValueError(f"unknown network type: {kind!r}")


def parameter_count(network: Network) -> int:
    """Learned numbers in the model: every weight+bias of a WeightVector, every enabled connection of
    a NEAT genome (its bias is a connection from the bias node, so it's counted the same way)."""
    if isinstance(network, NeatGenome):
        return len(network.enabled_connections)
    return len(network.weights)


def compiled(network: Network) -> dict:
    """A trained network as plain numbers a runtime that evaluates it itself can be built from -- the game
    core (docs/design/0009), in Python through PyO3 or in a browser through WebAssembly -- so no consumer
    re-implements NEAT's topological sort or the weight layout. Fixed topology: `{"kind": "layered",
    "weights": [...], "layer_sizes": [...]}` (`WeightVector`'s flat layout, tanh layers). Evolved graph:
    `{"kind": "graph", "encoding": [...]}` (`NeatGenome.graph_encoding()`, tanh on every node)."""
    if isinstance(network, NeatGenome):
        return {"kind": "graph", "encoding": network.graph_encoding()}
    return {"kind": "layered", "weights": list(network.weights), "layer_sizes": list(network.layer_sizes)}


def describe(network: Network) -> str:
    """Short human label for the model's shape: `11 → 16 → 3` for a fixed network, `11 → 7 hidden → 3 · 54
    conns` for an evolved graph (a NEAT genome has no layers, only how many hidden nodes it grew)."""
    if isinstance(network, NeatGenome):
        hidden, connections = network.complexity()
        return f"{network.num_inputs} → {hidden} hidden → {network.num_outputs} · {connections} conns"
    return " → ".join(str(n) for n in network.layer_sizes)
