"""A hand-coded Adam optimizer -- transformers are notoriously hard to train with plain SGD, and
building an adaptive optimizer is a small, well-understood amount of code on top of the autodiff
engine (it operates directly on .data/.grad, outside the differentiable graph, same as any real
framework's optimizer step)."""

from __future__ import annotations

import numpy as np
from autodiff import Tensor


class Adam:
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in parameters]
        self._v = [np.zeros_like(p.data) for p in parameters]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        for i, p in enumerate(self.parameters):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * p.grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (p.grad**2)
            m_hat = self._m[i] / (1 - self.beta1**self._t)
            v_hat = self._v[i] / (1 - self.beta2**self._t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()
