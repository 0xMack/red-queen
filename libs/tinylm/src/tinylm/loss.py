"""Language-model loss: cross-entropy between predicted next-token logits and the actual next
token, averaged over every position in the batch."""

from __future__ import annotations

import numpy as np

from autodiff import Tensor, softmax


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """`logits`: (batch, seq_len, vocab_size). `targets`: int array (batch, seq_len)."""
    batch, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(batch * seq_len, vocab_size)
    flat_targets = targets.reshape(-1)

    probs = softmax(flat_logits, axis=-1)
    rows = np.arange(batch * seq_len)
    correct_probs = probs[(rows, flat_targets)]  # probs[i, flat_targets[i]] for each row i
    return -(correct_probs.log().mean())
