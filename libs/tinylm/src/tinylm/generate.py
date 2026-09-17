"""Sampling from a trained model -- reads .data only, never calls .backward()."""

from __future__ import annotations

import numpy as np

from tinylm.model import TinyLM
from tinylm.tokenizer import CharTokenizer


def generate(
    model: TinyLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> str:
    ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        context = ids[-model.max_seq_len :]
        logits = model(np.array([context]))
        last_logits = logits.data[0, -1] / temperature
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()
        next_id = rng.choice(len(probs), p=probs)
        ids.append(int(next_id))
    return tokenizer.decode(ids)
