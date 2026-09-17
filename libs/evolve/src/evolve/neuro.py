"""A weight-vector genome -- neuroevolution/ES, the first representation (docs/design/0003
phase 5) that isn't program-shaped at all. `WeightVector` is the flattened weights+biases of a
small feedforward network; evolving it is literally Evolution Strategies applied to a policy
(docs/design/0003 "the case where RL and EA are the same algorithm").
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace


def _param_count(layer_sizes: tuple[int, ...]) -> int:
    return sum(
        layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1] for i in range(len(layer_sizes) - 1)
    )


def _forward(weights: Sequence[float], layer_sizes: tuple[int, ...], observation: Sequence[float]) -> list[float]:
    """A tanh-activated feedforward pass. Every layer (including the output) is tanh-squashed --
    convenient here since it keeps actions bounded to [-1, 1] with no separate output activation
    to choose."""
    activations = list(observation)
    offset = 0
    for i in range(len(layer_sizes) - 1):
        in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
        next_activations = []
        for o in range(out_size):
            total = weights[offset + in_size * out_size + o]  # bias
            for k in range(in_size):
                total += weights[offset + o * in_size + k] * activations[k]
            next_activations.append(math.tanh(total))
        offset += in_size * out_size + out_size
        activations = next_activations
    return activations


@dataclass(frozen=True, slots=True)
class WeightVector:
    """A small feedforward network's weights, flattened to one tuple: for each layer, its
    `out_size * in_size` weights followed by its `out_size` biases, in layer order."""

    weights: tuple[float, ...]
    layer_sizes: tuple[int, ...]

    def forward(self, observation: Sequence[float]) -> list[float]:
        return _forward(self.weights, self.layer_sizes, observation)

    def act(self, observation: Sequence[float]) -> float:
        """Convenience for single-output policies (a bounded scalar action, e.g. reach1d) --
        returns just the first output. Multi-output policies (e.g. discrete action selection via
        argmax, e.g. games.snake) should call forward() directly."""
        return self.forward(observation)[0]

    def l2_norm(self) -> float:
        """A complexity/regularization measure for ParetoSelection -- "simpler" here means
        smaller weights, not fewer of them (the network's shape is fixed, unlike a GP genome's
        size)."""
        return math.sqrt(sum(w * w for w in self.weights))


def random_weight_vector(
    layer_sizes: tuple[int, ...], rng: random.Random, scale: float = 1.0
) -> WeightVector:
    n = _param_count(layer_sizes)
    weights = tuple(rng.uniform(-scale, scale) for _ in range(n))
    return WeightVector(weights=weights, layer_sizes=layer_sizes)


class GaussianMutation:
    """ES-style variation: perturbs a parent's weights with independent Gaussian noise. No
    crossover -- unlike swapping GP instructions/subtrees, averaging or splicing two networks'
    weights doesn't generally combine their behavior (the same weight plays a different role
    depending on everything around it), so standard neuroevolution/ES practice is mutation-only.
    """

    def __init__(self, sigma: float = 0.1):
        self._sigma = sigma

    def vary(self, parents: list[WeightVector], rng: random.Random) -> WeightVector:
        parent = parents[0]
        new_weights = tuple(w + rng.gauss(0.0, self._sigma) for w in parent.weights)
        return replace(parent, weights=new_weights)
