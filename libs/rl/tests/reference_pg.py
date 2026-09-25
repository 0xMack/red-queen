"""Policy gradients, re-derived independently (docs/design/0010 Phase 3): the oracle for `pg::pg_gradients` and
`pg::gae`.

The loss is written as the formulas read -- log-softmax or a Gaussian log-density, the probability ratio, PPO's
`min(r A, clip(r) A)`, an entropy bonus -- on `libs/autodiff` Tensors, and differentiated by it; nothing is taken from
the Rust. GAE is the textbook backward recursion in plain Python. The Rust must agree to ~1e-12.
"""

from __future__ import annotations

import math

import numpy as np
from autodiff import Tensor
from reference_nn import split


def policy_loss_and_gradients(
    layer_sizes: list[int],
    params: np.ndarray,
    log_std: float | None,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip: float | None,
    entropy_coef: float,
) -> tuple[float, np.ndarray]:
    """(loss, gradients: the network's, then log_std's) for a tanh MLP policy."""
    layers = [(Tensor(w), Tensor(b)) for w, b in split(layer_sizes, params)]
    x = Tensor(observations)
    for i, (w, b) in enumerate(layers):
        x = x.matmul(w.transpose()) + b
        if i < len(layers) - 1:
            x = x.tanh()
    n = len(actions)
    rows = np.arange(n)
    if log_std is None:
        shifted = x - Tensor(x.data.max(axis=1, keepdims=True))  # a constant shift: log-softmax is invariant to it
        log_probs_all = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
        log_p = log_probs_all[(rows, actions.astype(int))]
        entropy = -(log_probs_all.exp() * log_probs_all).sum(axis=1)
        std_param = None
    else:
        std_param = Tensor(np.array([log_std]))
        mean = x[(rows, np.zeros(n, dtype=int))]
        sigma = std_param.exp()
        z = (Tensor(actions) - mean) / sigma
        log_p = z * z * -0.5 - std_param - 0.5 * math.log(2 * math.pi)
        entropy = std_param + 0.5 * math.log(2 * math.pi * math.e)
    if clip is None:
        surrogate = log_p * Tensor(advantages)
    else:
        ratio = (log_p - Tensor(old_log_probs)).exp()
        # min(r A, clip(r) A): the clipped term is a constant where it's the smaller one, so pick per sample
        clipped_ratio = np.clip(ratio.data, 1 - clip, 1 + clip)
        use_unclipped = ratio.data * advantages <= clipped_ratio * advantages
        surrogate = ratio * Tensor(advantages * use_unclipped) + Tensor(clipped_ratio * advantages * ~use_unclipped)
    loss = surrogate.sum() * (-1.0 / n) - entropy.sum() * (entropy_coef / (n if log_std is None else 1))
    loss.backward()
    grads = np.concatenate([np.concatenate([w.grad.ravel(), b.grad]) for w, b in layers])
    if std_param is not None:
        grads = np.concatenate([grads, std_param.grad])
    return float(loss.data), grads


def gae(rewards, values, next_values, done, cut, gamma, lam) -> tuple[list[float], list[float]]:
    """Generalized advantage estimation, as Schulman et al. (2016) define it, one episode segment at a time."""
    advantages = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if cut[t]:
            running = 0.0
        next_value = 0.0 if done[t] else next_values[t]
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, [a + v for a, v in zip(advantages, values, strict=True)]
