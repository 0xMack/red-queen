"""The DQN update, re-derived independently (docs/design/0010 Phase 2): the oracle for `dqn::td_gradients`.

Targets are plain numpy (they're constants to the update: no gradient flows through them); the prediction, the
Huber loss and every gradient come from `libs/autodiff`. Plain networks and dueling ones (trunk, value head,
advantage head, `Q = V + A - mean(A)`), with the target network or -- Double DQN -- the online one choosing the next
action. The Rust core must agree to ~1e-12.
"""

from __future__ import annotations

import numpy as np
from autodiff import Tensor
from reference_nn import split


def _mlp(x: Tensor, layers: list[tuple[Tensor, Tensor]], activations: list[str]) -> Tensor:
    for (w, b), activation in zip(layers, activations, strict=True):
        x = x.matmul(w.transpose()) + b
        if activation == "relu":
            x = x.relu()
    return x


def q_values(
    parts: list[list[Tensor]], inputs: int, hidden: list[int], actions: int, dueling: bool, x: Tensor
) -> Tensor:
    """`parts` are per-sub-network lists of (weights, biases) Tensor pairs, as `layers()` builds them."""
    if not dueling:
        return _mlp(x, parts[0], ["relu"] * len(hidden) + ["linear"])
    h = _mlp(x, parts[0], ["relu"] * len(hidden))
    value = _mlp(h, parts[1], ["linear"])
    advantage = _mlp(h, parts[2], ["linear"])
    return value + advantage - advantage.mean(axis=1, keepdims=True)


def layers(params: list[np.ndarray], inputs: int, hidden: list[int], actions: int, dueling: bool) -> list[list]:
    """Flat parameter vectors (the Rust `QNet::parts()` layout) -> per-sub-network (weights, biases) Tensors."""
    if not dueling:
        shapes = [[inputs, *hidden, actions]]
    else:
        shapes = [[inputs, *hidden], [hidden[-1], 1], [hidden[-1], actions]]
    return [[(Tensor(w), Tensor(b)) for w, b in split(s, p)] for s, p in zip(shapes, params, strict=True)]


def td_gradients(
    inputs: int,
    hidden: list[int],
    actions: int,
    dueling: bool,
    double: bool,
    online: list[np.ndarray],
    target: list[np.ndarray],
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, list[np.ndarray], np.ndarray, float]:
    """(loss, gradients per sub-network, td errors, mean Q(s, a)) for one minibatch."""
    observations, taken, returns, discounts, next_observations, weights = batch
    size = len(taken)
    rows = np.arange(size)

    def q(params: list[np.ndarray], x: np.ndarray) -> Tensor:
        return q_values(layers(params, inputs, hidden, actions, dueling), inputs, hidden, actions, dueling, Tensor(x))

    next_target = q(target, next_observations).data
    chooser = q(online, next_observations).data if double else next_target
    y = returns + discounts * next_target[rows, np.argmax(chooser, axis=1)]  # argmax: first of ties, like Rust

    parts = layers(online, inputs, hidden, actions, dueling)
    predicted = q_values(parts, inputs, hidden, actions, dueling, Tensor(observations))[(rows, taken)]
    delta = Tensor(y) - predicted
    inside = np.abs(delta.data) <= 1.0
    huber = delta * delta * (0.5 * inside) + (delta * np.sign(delta.data) - 0.5) * (~inside)
    loss = (huber * weights).sum() * (1.0 / size)
    loss.backward()
    grads = [np.concatenate([np.concatenate([w.grad.ravel(), b.grad]) for w, b in part]) for part in parts]
    return float(loss.data), grads, delta.data, float(predicted.data.mean())
