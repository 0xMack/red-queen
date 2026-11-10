"""The training stream (docs/design/0024 Decision 3): images with the token sequences of the clean SVGs behind them.

`synthetic` is generated on seeds from `GENERATED_BASE` up, disjoint from every split (train 0-95, held-out 10,000+, validate
20,000-20,399), and each image is corrupted afresh by `corrupt.v1` at a random severity, so no corrupted image repeats. A
stream is a pure function of its seed: sample `i` is always the same image and tokens, so a run is reproducible however
many workers read it.

The stream itself is plain Python and numpy; `svgformer.train.SyntheticBatches` wraps it for a `DataLoader`.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from arena.copycat_sets import SEVERITIES, SIZE, corrupt_pixels, synthetic
from PIL import Image

from svgformer import tokens as T

GENERATED_BASE = 1_000_000
STREAM_STRIDE = 100_000_000  # stream seeds this far apart never share a generated seed


@dataclass(frozen=True)
class Sample:
    image: np.ndarray  # (3, SIZE, SIZE) uint8
    tokens: list[int]  # BOS ... END
    seed: int
    severity: str


def png_to_array(png: bytes) -> np.ndarray:
    """A PNG as (3, H, W) uint8."""
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).transpose(2, 0, 1).copy()


def synthetic_sample(stream_seed: int, i: int, severities: tuple[str, ...] = SEVERITIES) -> Sample:
    seed = GENERATED_BASE + stream_seed * STREAM_STRIDE + i
    svg, _ = synthetic(seed)
    severity = random.Random(f"svgformer:{seed}").choice(severities)
    pixels, _ = corrupt_pixels(svg, seed, severity)
    image = np.asarray(pixels.convert("RGB")).transpose(2, 0, 1).copy()
    return Sample(image, T.encode(svg, SIZE).tokens, seed, severity)


def collate(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Images (B, 3, H, W) uint8 and tokens (B, N) int64, padded with PAD to the longest."""
    n = max(len(s.tokens) for s in samples)
    tokens = np.full((len(samples), n), T.PAD, dtype=np.int64)
    for row, s in enumerate(samples):
        tokens[row, : len(s.tokens)] = s.tokens
    return np.stack([s.image for s in samples]), tokens
