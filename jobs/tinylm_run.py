"""Train TinyLM on Alice's Adventures in Wonderland and save a checkpoint (docs/design/0004, 0009).

Same model and schedule as notebooks/0006-tinylm-from-scratch.ipynb (the numbers the Transformers
Learn chapter cites), except that the last 5% of the corpus is held out: never trained on, used for a
validation loss here and, by jobs/publish_models.py, as the text an exported package is checked on
(top-1 next-character agreement with this checkpoint).

Writes jobs/run-data/tinylm/<name>.npz + .json. Not a telemetry run (tinylm isn't wired into
telemetry, doc 0004); the model store's catalog is what makes it visible to the site.

Run with: uv run python jobs/tinylm_run.py [name]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from run_context import RUN_DATA_DIR
from tinylm import Adam, CharTokenizer, TinyLM, cross_entropy, save

CORPUS = Path(__file__).parents[1] / "libs" / "tinylm" / "data" / "alice.txt"
HELD_OUT_FRACTION = 0.05

SEQ_LEN = 32
BATCH_SIZE = 32
STEPS = 3000
LR = 3e-3
MODEL = {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_hidden": 128}


def split_corpus() -> tuple[CharTokenizer, np.ndarray, np.ndarray]:
    """(tokenizer over the whole corpus, training ids, held-out ids) -- the vocabulary covers the held-out
    text too, so every held-out character is encodable."""
    text = CORPUS.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = np.array(tokenizer.encode(text))
    cut = int(len(ids) * (1 - HELD_OUT_FRACTION))
    return tokenizer, ids[:cut], ids[cut:]


def windows(ids: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    starts = rng.integers(0, len(ids) - SEQ_LEN - 1, size=count)
    x = np.stack([ids[s : s + SEQ_LEN] for s in starts])
    y = np.stack([ids[s + 1 : s + SEQ_LEN + 1] for s in starts])
    return x, y


def main(name: str = "alice-v1") -> None:
    tokenizer, train, held_out = split_corpus()
    rng = np.random.default_rng(0)
    model = TinyLM(vocab_size=tokenizer.vocab_size, max_seq_len=SEQ_LEN, rng=rng, **MODEL)
    optimizer = Adam(model.parameters(), lr=LR)
    parameters = sum(p.data.size for p in model.parameters())
    print(f"{parameters:,} parameters, {len(train):,} training / {len(held_out):,} held-out characters")

    started = time.perf_counter()
    losses = []
    for step in range(STEPS):
        x, y = windows(train, BATCH_SIZE, rng)
        loss = cross_entropy(model(x), y)
        losses.append(float(loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"{step:5d}  loss {loss.data:.3f}  {time.perf_counter() - started:.0f}s")
    train_s = time.perf_counter() - started

    x, y = windows(held_out, 256, np.random.default_rng(1))
    val_loss = float(cross_entropy(model(x), y).data)
    print(f"trained in {train_s:.0f}s; train loss (last 50) {np.mean(losses[-50:]):.3f}, held-out loss {val_loss:.3f}")
    path = RUN_DATA_DIR / "tinylm" / name
    save(
        model,
        tokenizer,
        path,
        name=name,
        corpus="libs/tinylm/data/alice.txt",
        held_out_fraction=HELD_OUT_FRACTION,
        training={"steps": STEPS, "batch_size": BATCH_SIZE, "lr": LR, "seconds": round(train_s, 1)},
        train_loss=round(float(np.mean(losses[-50:])), 4),
        held_out_loss=round(val_loss, 4),
        parameters=parameters,
    )
    print(f"wrote {path}.npz/.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
