"""Grammar-constrained decoding (docs/design/0024 Decision 2): at every step each sequence's logits are masked to the
tokens its grammar state allows, so every sample decodes into an SVG the subset parses. Greedy (`temperature=0`) or
sampled, batched, with the decoder's KV cache.
"""

from __future__ import annotations

import numpy as np
import torch

from svgformer import grammar
from svgformer import tokens as T
from svgformer.model import SvgFormer

_PAD_ONLY = np.zeros(T.VOCAB, dtype=bool)
_PAD_ONLY[T.PAD] = True


@torch.no_grad()
def generate(
    model: SvgFormer,
    images: torch.Tensor,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Token sequences (BOS ... END, or cut off at `max_tokens`) for images (B, 3, H, W) in [0, 1]."""
    max_tokens = min(max_tokens or model.config.max_tokens, model.config.max_tokens)
    device = images.device
    b = images.shape[0]
    memories = model.memories(model.encode(images))
    caches: list[list[torch.Tensor | None]] = [[None, None] for _ in model.decoder]
    states = [grammar.advance(grammar.START, T.BOS)] * b
    sequences = [[T.BOS] for _ in range(b)]
    done = [False] * b
    token = torch.full((b,), T.BOS, dtype=torch.long, device=device)
    for position in range(max_tokens - 1):
        logits = model.step(token, position, memories, caches).float()
        allowed = np.stack([_PAD_ONLY if done[i] else grammar.mask(states[i]) for i in range(b)])
        logits = logits.masked_fill(~torch.from_numpy(allowed).to(device), float("-inf"))
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
        else:
            token = logits.argmax(dim=-1)
        chosen = token.tolist()
        for i, t in enumerate(chosen):
            if done[i]:
                continue
            sequences[i].append(t)
            states[i] = grammar.advance(states[i], t)
            done[i] = t == T.END
        if all(done):
            break
    return sequences


def to_images(rgb8: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """(B, 3, H, W) uint8 -> float in [0, 1] on `device`."""
    return torch.from_numpy(rgb8).to(device).float() / 255
