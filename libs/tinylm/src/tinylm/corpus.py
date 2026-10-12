"""The training corpus (docs/design/0004): Alice's Adventures in Wonderland, its last 5% held out -- never trained on,
used for a validation loss and as the text an exported package is checked on. Shared by training (the trainer's
`tinylm`) and publishing, so both cut it the same way."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tinylm.tokenizer import CharTokenizer

CORPUS = Path(__file__).parents[2] / "data" / "alice.txt"
CORPUS_NAME = "libs/tinylm/data/alice.txt"
HELD_OUT_FRACTION = 0.05


def split_corpus() -> tuple[CharTokenizer, np.ndarray, np.ndarray]:
    """(tokenizer over the whole corpus, training ids, held-out ids) -- the vocabulary covers the held-out
    text too, so every held-out character is encodable."""
    text = CORPUS.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = np.array(tokenizer.encode(text))
    cut = int(len(ids) * (1 - HELD_OUT_FRACTION))
    return tokenizer, ids[:cut], ids[cut:]
