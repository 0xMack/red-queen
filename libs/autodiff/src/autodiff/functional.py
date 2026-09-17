"""Composite ops built from Tensor primitives -- their backward pass comes for free from the
graph, no hand-derived rule needed here."""

from __future__ import annotations

import numpy as np

from autodiff.tensor import Tensor


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax: subtracting the max before exponentiating doesn't change the
    result (or the gradient) mathematically, only the numerics -- so it's done with a plain,
    ungraphed NumPy constant rather than a tracked op."""
    shifted = x - Tensor(np.max(x.data, axis=axis, keepdims=True))
    exp = shifted.exp()
    return exp / exp.sum(axis=axis, keepdims=True)
