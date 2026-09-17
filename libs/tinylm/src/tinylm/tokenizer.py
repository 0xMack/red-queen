"""Character-level tokenization (docs/design/0004: simplest option, no BPE implementation needed
as a prerequisite)."""

from __future__ import annotations

from collections.abc import Iterable


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = dict(enumerate(chars))

    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self._itos[i] for i in ids)
