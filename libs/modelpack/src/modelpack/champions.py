"""Every kind of trained champion a run stores, loaded from its artifact (docs/design/0010 Decision 5).

Evolved networks (`WeightVector`, `NeatGenome`) keep their loader in `evolve.network_from_json`; reinforcement
learning adds kinds `evolve` must never know about (it doesn't depend on `rl`), so the one loader for *all* of them
lives here -- `modelpack` already depends on every network kind it exports.

- `qtable` (tabular Q-learning / SARSA, `libs/rl`): one row of action values per discrete state.
- `mlp` (DQN, `libs/rl`): a dense network with per-layer activations -- unlike `evolve`'s `WeightVector`, which is
  tanh on every layer, Q-values need ReLU hidden layers and a linear output.

`load_champion(text)` returns something with `forward(observation) -> outputs`, which is all evaluation and export
need. A kind that has no policy to package (the RL pipeline's `random` agent) raises `UnsupportedChampion`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from evolve.networks import Network, describe, network_from_json, parameter_count


class UnsupportedChampion(ValueError):
    """A stored champion with no policy this package can load or export."""


@dataclass(frozen=True)
class QTable:
    """A tabular policy: `values[state * actions + action]`, the state being the row a `Discretizer` computes
    (`libs/rl/rust/core/src/tabular.rs`). Only a binary discretizer can be exported: the row is then the dot product
    of `observation != 0` with `[1, 2, 4, ...]`."""

    algorithm: str
    discretizer: dict[str, Any]
    actions: dict[str, Any]
    values: tuple[float, ...]

    @property
    def num_actions(self) -> int:
        return len(self.actions["values"]) if self.actions["kind"] == "continuous" else int(self.actions["count"])

    @property
    def states(self) -> int:
        return len(self.values) // self.num_actions

    @property
    def num_inputs(self) -> int:
        if self.discretizer["kind"] == "binary":
            return int(self.discretizer["bits"])
        return len(self.discretizer["dims"])

    def index(self, observation: Sequence[float]) -> int:
        if self.discretizer["kind"] == "binary":
            return sum(1 << i for i, v in enumerate(observation) if v != 0)
        index, radix = 0, 1
        for value, (low, high, n) in zip(observation, self.discretizer["dims"], strict=True):
            t = (value - low) / (high - low) * n
            index += (0 if t < 0 else min(int(t), n - 1)) * radix
            radix *= n
        return index

    def forward(self, observation: Sequence[float]) -> list[float]:
        start = self.index(observation) * self.num_actions
        return list(self.values[start : start + self.num_actions])

    def visited_states(self) -> int:
        """Rows that differ from the table's initial value -- states the agent actually learned about."""
        initial = self.values[0] if self.values else 0.0
        rows = (self.values[s * self.num_actions : (s + 1) * self.num_actions] for s in range(self.states))
        return sum(1 for row in rows if any(v != initial for v in row))

    @staticmethod
    def from_dict(data: dict[str, Any]) -> QTable:
        table = QTable(
            algorithm=str(data.get("algorithm", "q_learning")),
            discretizer=dict(data["discretizer"]),
            actions=dict(data["actions"]),
            values=tuple(float(v) for v in data["values"]),
        )
        if table.num_actions <= 0 or len(table.values) % table.num_actions:
            raise ValueError(f"a {table.num_actions}-action table can't hold {len(table.values)} values")
        return table


ACTIVATIONS = {"linear": lambda x: x, "relu": lambda x: np.maximum(x, 0.0), "tanh": np.tanh}


@dataclass(frozen=True)
class MlpPolicy:
    """A dense network (`libs/rl/rust/core/src/nn.rs`): `layer_sizes` (inputs first), one activation per layer of
    weights, and `params` in `evolve.WeightVector`'s flat layout -- per layer, `out * in` weights (row = output unit),
    then `out` biases. Its outputs are action values an interface's argmax decodes."""

    algorithm: str
    layer_sizes: tuple[int, ...]
    activations: tuple[str, ...]
    params: tuple[float, ...]

    def layers(self) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """(weights[out, in], biases[out], activation) per layer."""
        flat = np.asarray(self.params, dtype=np.float64)
        layers, at = [], 0
        for (n_in, n_out), activation in zip(pairwise(self.layer_sizes), self.activations, strict=True):
            w = flat[at : at + n_in * n_out].reshape(n_out, n_in)
            b = flat[at + n_in * n_out : at + n_out * (n_in + 1)]
            layers.append((w, b, activation))
            at += n_out * (n_in + 1)
        return layers

    def forward(self, observation: Sequence[float]) -> list[float]:
        x = np.asarray(observation, dtype=np.float64)
        for w, b, activation in self._split():
            x = ACTIVATIONS[activation](w @ x + b)
        return x.tolist()

    def _split(self) -> list[tuple[np.ndarray, np.ndarray, str]]:
        # frozen, but splitting the flat vector is pure: do it once (evaluation calls forward ~10^5 times)
        cached = self.__dict__.get("_layers")
        if cached is None:
            cached = self.layers()
            object.__setattr__(self, "_layers", cached)
        return cached

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MlpPolicy:
        policy = MlpPolicy(
            algorithm=str(data.get("algorithm", "dqn")),
            layer_sizes=tuple(int(n) for n in data["layer_sizes"]),
            activations=tuple(str(a) for a in data["activations"]),
            params=tuple(float(v) for v in data["params"]),
        )
        expected = sum(n_out * (n_in + 1) for n_in, n_out in pairwise(policy.layer_sizes))
        if len(policy.activations) != len(policy.layer_sizes) - 1 or len(policy.params) != expected:
            raise ValueError(f"layers {policy.layer_sizes} need {expected} parameters and one activation per layer")
        unknown = set(policy.activations) - set(ACTIVATIONS)
        if unknown:
            raise ValueError(f"unknown activation(s) {sorted(unknown)}")
        return policy


Champion = Network | QTable | MlpPolicy


def load_champion(text: str) -> Champion:
    """The champion stored in a run's artifact: an evolved network, a Q-table or a Q-network."""
    data = json.loads(text)
    kind = data.get("type") if isinstance(data, dict) else None
    if kind == "qtable":
        return QTable.from_dict(data)
    if kind == "mlp":
        return MlpPolicy.from_dict(data)
    if kind == "random":
        raise UnsupportedChampion("a random agent has no policy to load (it's the RL pipeline's smoke test)")
    try:
        return network_from_json(text)
    except (KeyError, TypeError, ValueError) as e:
        raise UnsupportedChampion(f"unknown champion type {kind!r}: {e}") from e


def describe_champion(champion: Champion) -> str:
    """Short label for the model's shape, like `evolve.networks.describe`."""
    if isinstance(champion, QTable):
        return f"table {champion.states} states x {champion.num_actions} actions"
    if isinstance(champion, MlpPolicy):
        return " → ".join(str(n) for n in champion.layer_sizes)
    return describe(champion)


def champion_parameters(champion: Champion) -> int:
    if isinstance(champion, QTable):
        return len(champion.values)
    if isinstance(champion, MlpPolicy):
        return len(champion.params)
    return parameter_count(champion)
