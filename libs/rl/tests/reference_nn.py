"""The gradient oracle (docs/design/0010 Decision 1): the Rust core's network and optimizer, re-derived independently.

Gradients come from `libs/autodiff`'s reverse-mode engine (the project's own, built for tinylm), never from the
Rust backprop being checked; Adam is written out from the paper with numpy. The Rust core must agree to ~1e-12 --
not bit for bit, because numpy's matmul sums in a different order.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from autodiff import Tensor


def split(layer_sizes: list[int], params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """`evolve.WeightVector`'s flat layout -> per layer (weights[out, in], biases[out])."""
    layers, at = [], 0
    for n_in, n_out in pairwise(layer_sizes):
        w = params[at : at + n_in * n_out].reshape(n_out, n_in)
        b = params[at + n_in * n_out : at + n_out * (n_in + 1)]
        layers.append((w, b))
        at += n_out * (n_in + 1)
    return layers


def forward_backward(
    layer_sizes: list[int],
    activations: list[str],
    params: np.ndarray,
    inputs: np.ndarray,
    output_grad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(outputs, dL/dparams) where L = sum(outputs * output_grad) -- so dL/doutputs is exactly `output_grad`."""
    tensors = [(Tensor(w), Tensor(b)) for w, b in split(layer_sizes, params)]
    x = Tensor(inputs)
    for (w, b), activation in zip(tensors, activations, strict=True):
        x = x.matmul(w.transpose()) + b
        if activation == "relu":
            x = x.relu()
        elif activation == "tanh":
            x = x.tanh()
        else:
            assert activation == "linear", activation
    loss = (x * Tensor(output_grad)).sum()
    loss.backward()
    grads = np.concatenate([np.concatenate([w.grad.ravel(), b.grad]) for w, b in tensors])
    return x.data, grads


def adam_steps(
    params: np.ndarray,
    gradients: list[np.ndarray],
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Kingma & Ba (2015), Algorithm 1, one step per gradient."""
    params = params.copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    for t, g in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return params
