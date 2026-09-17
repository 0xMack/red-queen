"""The full model: token + positional embeddings, a stack of transformer blocks, a final layer
norm, and an output projection to vocabulary logits."""

from __future__ import annotations

import numpy as np

from autodiff import Tensor
from tinylm.layers import Embedding, LayerNorm, Linear, TransformerBlock


class TinyLM:
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_hidden: int,
        rng: np.random.Generator,
    ):
        self.max_seq_len = max_seq_len
        self.token_embedding = Embedding(vocab_size, d_model, rng)
        self.position_embedding = Embedding(max_seq_len, d_model, rng)
        self.blocks = [TransformerBlock(d_model, n_heads, d_hidden, rng) for _ in range(n_layers)]
        self.ln_final = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def __call__(self, token_ids: np.ndarray) -> Tensor:
        """`token_ids`: int array of shape (batch, seq_len). Returns logits (batch, seq_len, vocab_size)."""
        _batch, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_embedding(token_ids) + self.position_embedding(np.arange(seq_len))
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)

    def parameters(self) -> list[Tensor]:
        params = self.token_embedding.parameters() + self.position_embedding.parameters()
        for block in self.blocks:
            params += block.parameters()
        return params + self.ln_final.parameters() + self.head.parameters()
