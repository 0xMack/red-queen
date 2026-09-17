"""A reverse-mode automatic differentiation engine, built from scratch (docs/design/0004).

Unlike the canonical scalar-valued teaching example (Karpathy's `micrograd`, one graph node per
float), `Tensor` here wraps a NumPy array: a transformer even at toy scale pushes thousands of
values through matrix multiplications, and a graph node per scalar would make Python object
overhead dominate. NumPy is the array-arithmetic substrate underneath -- the same role
`random.Random` plays for `evolve` -- while the actual autodiff mechanism (the graph, the backward
rule for each op, `.backward()`) is what's built here.
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sums `grad` back down to `shape`, undoing whatever NumPy broadcasting happened in the
    forward pass (e.g. adding a `(features,)` bias to a `(batch, features)` tensor) -- forward
    broadcasting implicitly sums contributions from every broadcasted position, so gradients need
    to sum back over exactly those positions to be correct."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A node in the computation graph: a NumPy array, its accumulated gradient, and (if it was
    computed from other Tensors) a `_backward` closure that propagates `self.grad` to its parents.
    """

    def __init__(self, data, _parents: tuple[Tensor, ...] = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def _as_tensor(value) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value)

    def __add__(self, other) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other) -> Tensor:
        return self + (-self._as_tensor(other))

    def __rsub__(self, other) -> Tensor:
        return self._as_tensor(other) + (-self)

    def __mul__(self, other) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power: float) -> Tensor:
        out = Tensor(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other) -> Tensor:
        return self * self._as_tensor(other) ** -1.0

    def __rtruediv__(self, other) -> Tensor:
        return self._as_tensor(other) * self**-1.0

    def matmul(self, other: Tensor) -> Tensor:
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    __matmul__ = matmul

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def exp(self) -> Tensor:
        exp_data = np.exp(self.data)
        out = Tensor(exp_data, (self,), "exp")

        def _backward():
            self.grad += exp_data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        axes_arg = axes or None
        out = Tensor(self.data.transpose(axes_arg), (self,), "transpose")

        def _backward():
            if axes_arg is None:
                self.grad += out.grad.transpose()
            else:
                self.grad += out.grad.transpose(tuple(np.argsort(axes_arg)))

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Seeds this tensor's gradient with ones and propagates gradients to every ancestor, in
        reverse topological order -- each node's `_backward` runs exactly once, after every node
        that depends on it has already contributed its share of the gradient."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._parents:
                    build(parent)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)
