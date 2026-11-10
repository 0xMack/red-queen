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


_COMMAND_OF = {T.COMMAND_ID[c]: c for c in T.COMMANDS}
REPEAT_WINDOW = 4  # a path segment may not repeat any of this many before it
STALL_SEGMENTS, STALL_SPAN = (
    6,
    16,
)  # a path whose last 6 segments all end within 16 grid steps (1.6 px at 128) has stalled


class _Segments:
    """One sequence's path segments as they are written, for `no_repeat`. Greedy decoding falls into loops on long artwork
    paths (0024 E2): `L 640 810` written until the length cap, and a path cut off there is dropped whole. So the number
    token that would complete a segment going nowhere (every point of it the current point) or repeating one of the last
    `REPEAT_WINDOW` exactly is banned, and a path that has stalled anyway (its last `STALL_SEGMENTS` segments ending within
    `STALL_SPAN` grid steps of each other, drifting or alternating) is ended: `PATH_END` is the only token allowed."""

    def __init__(self) -> None:
        self.in_path = False
        self.command: str | None = None
        self.numbers: list[int] = []
        self.current: tuple[int, int] | None = None
        self.recent: list[tuple[str, tuple[int, ...]]] = []
        self.ends: list[tuple[int, int]] = []

    def push(self, token: int) -> None:
        if token == T.TYPE_ID["path"]:
            self.in_path, self.command, self.current, self.recent, self.ends = True, None, None, [], []
        elif token == T.PATH_END or (T.NAMES[token] in T.TYPES):
            self.in_path, self.command = False, None
        elif self.in_path and token in _COMMAND_OF:
            self.command, self.numbers = _COMMAND_OF[token], []
        elif self.in_path and self.command and T.is_number(token):
            self.numbers.append(token)
            if len(self.numbers) == T.COMMAND_NUMBERS[self.command]:
                segment = (self.command, tuple(self.numbers))
                self.recent = [*self.recent, segment][-REPEAT_WINDOW:]
                self.current = (self.numbers[-2], self.numbers[-1])
                if self.command != "M":
                    self.ends = [*self.ends, self.current][-STALL_SEGMENTS:]
                self.command = None

    @property
    def stalled(self) -> bool:
        """Between segments of a path whose last `STALL_SEGMENTS` segments ended within `STALL_SPAN` of each other."""
        if not self.in_path or self.command is not None or len(self.ends) < STALL_SEGMENTS:
            return False
        xs, ys = [e[0] for e in self.ends], [e[1] for e in self.ends]
        return max(xs) - min(xs) <= STALL_SPAN and max(ys) - min(ys) <= STALL_SPAN

    def banned(self) -> list[int]:
        """Number tokens not allowed next (empty unless the next token completes a segment)."""
        if not self.in_path or self.command in (None, "M", "Z"):
            return []
        need = T.COMMAND_NUMBERS[self.command]
        if len(self.numbers) != need - 1:
            return []
        prefix = tuple(self.numbers)
        out = [s[1][-1] for s in self.recent if s[0] == self.command and s[1][:-1] == prefix]
        if self.current is not None:
            x, y = self.current
            if all(prefix[k] == (x if k % 2 == 0 else y) for k in range(len(prefix))):
                out.append(y)  # every point the current point: a segment of length zero
        return out


@torch.no_grad()
def generate(
    model: SvgFormer,
    images: torch.Tensor,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    no_repeat: bool = False,
) -> list[list[int]]:
    """Token sequences (BOS ... END, or cut off at `max_tokens`) for images (B, 3, H, W) in [0, 1]. With `no_repeat`, a
    path segment may not go nowhere or repeat one of the few before it (`_Segments`)."""
    max_tokens = min(max_tokens or model.config.max_tokens, model.config.max_tokens)
    device = images.device
    b = images.shape[0]
    memories = model.memories(model.encode(images))
    caches: list[list[torch.Tensor | None]] = [[None, None] for _ in model.decoder]
    states = [grammar.advance(grammar.START, T.BOS)] * b
    sequences = [[T.BOS] for _ in range(b)]
    done = [False] * b
    segments = [_Segments() for _ in range(b)] if no_repeat else None
    token = torch.full((b,), T.BOS, dtype=torch.long, device=device)
    for position in range(max_tokens - 1):
        logits = model.step(token, position, memories, caches).float()
        allowed = np.stack([_PAD_ONLY if done[i] else grammar.mask(states[i]) for i in range(b)])
        if segments is not None:
            for i in range(b):
                if done[i]:
                    continue
                if segments[i].stalled and allowed[i, T.PATH_END]:
                    allowed[i] = False
                    allowed[i, T.PATH_END] = True
                else:
                    allowed[i, segments[i].banned()] = False
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
            if segments is not None:
                segments[i].push(t)
            done[i] = t == T.END
        if all(done):
            break
    return sequences


def to_images(rgb8: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """(B, 3, H, W) uint8 -> float in [0, 1] on `device`."""
    return torch.from_numpy(rgb8).to(device).float() / 255
