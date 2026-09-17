"""Transformer building blocks, all implemented as autodiff.Tensor ops -- no manual backward pass
anywhere here; the autodiff engine (docs/design/0004, libs/autodiff) computes every gradient.

Each layer follows a small from-scratch convention (not torch.nn): a plain class holding its
learnable Tensors as attributes, callable via `__call__`, with a `parameters()` method returning a
flat list of every learnable Tensor (used by the optimizer).
"""

from __future__ import annotations

import numpy as np

from autodiff import Tensor, softmax


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(rng.uniform(-scale, scale, size=(in_features, out_features)))
        self.bias = Tensor(np.zeros(out_features))

    def __call__(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class Embedding:
    def __init__(self, num_embeddings: int, dim: int, rng: np.random.Generator):
        self.weight = Tensor(rng.normal(scale=0.02, size=(num_embeddings, dim)))

    def __call__(self, idx: np.ndarray) -> Tensor:
        return self.weight[idx]

    def parameters(self) -> list[Tensor]:
        return [self.weight]


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / (variance + self.eps) ** 0.5
        return normalized * self.gamma + self.beta

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]


def _causal_mask(seq_len: int) -> np.ndarray:
    """0 on/below the diagonal, a large negative number above it -- added to attention scores
    before softmax so a position can't attend to future positions."""
    return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = Linear(d_model, d_model, rng)
        self.key = Linear(d_model, d_model, rng)
        self.value = Linear(d_model, d_model, rng)
        self.out_proj = Linear(d_model, d_model, rng)

    def __call__(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.query(x), batch, seq_len)
        k = self._split_heads(self.key(x), batch, seq_len)
        v = self._split_heads(self.value(x), batch, seq_len)

        scores = q.matmul(k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        scores = scores + Tensor(_causal_mask(seq_len))
        weights = softmax(scores, axis=-1)
        out = weights.matmul(v)  # (batch, n_heads, seq_len, head_dim)
        return self.out_proj(self._merge_heads(out, batch, seq_len))

    def _split_heads(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        return x.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

    def parameters(self) -> list[Tensor]:
        return (
            self.query.parameters()
            + self.key.parameters()
            + self.value.parameters()
            + self.out_proj.parameters()
        )


class MLP:
    def __init__(self, d_model: int, d_hidden: int, rng: np.random.Generator):
        self.fc1 = Linear(d_model, d_hidden, rng)
        self.fc2 = Linear(d_hidden, d_model, rng)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())

    def parameters(self) -> list[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rng: np.random.Generator):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_hidden, rng)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def parameters(self) -> list[Tensor]:
        return (
            self.ln1.parameters() + self.attn.parameters() + self.ln2.parameters() + self.mlp.parameters()
        )
