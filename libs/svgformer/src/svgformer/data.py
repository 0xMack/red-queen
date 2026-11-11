"""The training stream (docs/design/0024 Decision 3): images with the token sequences of the clean SVGs behind them.

`synthetic` is generated on seeds from `GENERATED_BASE` up, disjoint from every split (train 0-95, held-out 10,000+, validate
20,000-20,399), and each image is corrupted afresh by `corrupt.v1` at a random severity, so no corrupted image repeats. A
stream is a pure function of its seed: sample `i` is always the same image and tokens, so a run is reproducible however
many workers read it.

With `artwork_share` > 0 the stream also draws the training artwork (docs/design/0024 E2): Iconify's openly licensed
icons, converted by `svgformer.build`, which never includes Twemoji or Flat Color Icons (the `icons` and `broad` sets).

The stream itself is plain Python and numpy; `svgformer.train.SyntheticBatches` wraps it for a `DataLoader`.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arena.copycat_sets import SEVERITIES, SIZE, corrupt_pixels, synthetic
from games.copycat import Svg
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


def synthetic_draw(stream_seed: int, i: int, severities: tuple[str, ...] = SEVERITIES) -> tuple[str, int, str]:
    """Sample `i` of a stream's `synthetic` before corruption: (clean SVG, corruption seed, severity)."""
    seed = GENERATED_BASE + stream_seed * STREAM_STRIDE + i
    svg, _ = synthetic(seed)
    return svg, seed, random.Random(f"svgformer:{seed}").choice(severities)


def synthetic_sample(stream_seed: int, i: int, severities: tuple[str, ...] = SEVERITIES) -> Sample:
    svg, seed, severity = synthetic_draw(stream_seed, i, severities)
    pixels, _ = corrupt_pixels(svg, seed, severity)
    image = np.asarray(pixels.convert("RGB")).transpose(2, 0, 1).copy()
    return Sample(image, T.encode(svg, SIZE).tokens, seed, severity)


ARTWORK_BASE = 3_000_000_000  # corruption seeds of artwork samples: past every generated seed a stream can reach


@dataclass(frozen=True)
class ArtworkPool:
    """The training artwork (`svgformer.build`'s `icons.jsonl`) that fits: drawn at `SIZE`, nothing clamped, at most
    `max_tokens` tokens. Only the qualifying lines' byte offsets are held, so a DataLoader worker reads an SVG when it
    needs one rather than keeping the whole file (200+ MB) in each worker's memory."""

    path: str
    offsets: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path, max_tokens: int) -> ArtworkPool:
        offsets, at = [], 0
        with open(path, "rb") as f:
            for line in f:
                r = json.loads(line)
                if r["size"] == SIZE and r["clamped"] == 0 and r["tokens"] <= max_tokens:
                    offsets.append(at)
                at += len(line)
        if not offsets:
            raise ValueError(f"no artwork in {path} fits {max_tokens} tokens")
        return cls(str(path), tuple(offsets))

    def __len__(self) -> int:
        return len(self.offsets)

    def svg(self, k: int) -> str:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[k])
            return json.loads(f.readline())["svg"]


def artwork_draw(
    pool: ArtworkPool, stream_seed: int, i: int, severities: tuple[str, ...] = SEVERITIES
) -> tuple[str, int, str]:
    """Sample `i` of a stream's artwork before corruption: (clean SVG, corruption seed, severity)."""
    seed = ARTWORK_BASE + stream_seed * STREAM_STRIDE + i
    rng = random.Random(f"svgformer-artwork:{seed}")
    svg = Svg(pool.svg(rng.randrange(len(pool)))).canonical()  # canonical: one shape a line, as corrupt.v1 drifts them
    return svg, seed, rng.choice(severities)


def artwork_sample(pool: ArtworkPool, stream_seed: int, i: int, severities: tuple[str, ...] = SEVERITIES) -> Sample:
    """Sample `i` of a stream's artwork: a pool icon drawn at random (with replacement) and corrupted afresh, as `icons`
    and `broad` are, so the same icon is a different image each time it is drawn."""
    svg, seed, severity = artwork_draw(pool, stream_seed, i, severities)
    pixels, _ = corrupt_pixels(svg, seed, severity)
    image = np.asarray(pixels.convert("RGB")).transpose(2, 0, 1).copy()
    return Sample(image, T.encode(svg, SIZE).tokens, seed, severity)


def mixed_sample(
    stream_seed: int,
    i: int,
    severities: tuple[str, ...] = SEVERITIES,
    pool: ArtworkPool | None = None,
    artwork_share: float = 0.0,
) -> Sample:
    """Sample `i` of the training stream: artwork with probability `artwork_share`, else `synthetic`. With no artwork it
    is exactly `synthetic_sample`, so runs from before artwork was mixed in are reproduced."""
    if is_artwork(stream_seed, i, pool, artwork_share):
        return artwork_sample(pool, stream_seed, i, severities)
    return synthetic_sample(stream_seed, i, severities)


def is_artwork(stream_seed: int, i: int, pool: ArtworkPool | None, artwork_share: float) -> bool:
    return (
        pool is not None
        and artwork_share > 0
        and random.Random(f"svgformer-mix:{stream_seed}:{i}").random() < artwork_share
    )


def mixed_draw(
    stream_seed: int,
    i: int,
    severities: tuple[str, ...] = SEVERITIES,
    pool: ArtworkPool | None = None,
    artwork_share: float = 0.0,
) -> tuple[str, int, str]:
    """`mixed_sample`'s choice before corruption: (clean SVG, corruption seed, severity)."""
    if is_artwork(stream_seed, i, pool, artwork_share):
        return artwork_draw(pool, stream_seed, i, severities)
    return synthetic_draw(stream_seed, i, severities)


def collate(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Images (B, 3, H, W) uint8 and tokens (B, N) int64, padded with PAD to the longest."""
    n = max(len(s.tokens) for s in samples)
    tokens = np.full((len(samples), n), T.PAD, dtype=np.int64)
    for row, s in enumerate(samples):
        tokens[row, : len(s.tokens)] = s.tokens
    return np.stack([s.image for s in samples]), tokens
