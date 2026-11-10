"""A trained svgformer as a Copy Cat entrant (docs/design/0024 P3): greedy, grammar-constrained decoding, then the tracer's
polish (`games.copycat.polish`) against the input. `pick` pairs it with the MDL tracer, one or the other per image.

The pick (0024, "The model or the tracer, per image"): both SVGs are drawn over the background the tracer fits, and the
tracer's is taken only when its squared error against the input is under `1 / PICK_MARGIN` of the model's. Taking the
plain better fit is wrong under corruption: the tracer also draws the drift, shading and clutter that corruption adds, so
it fits a corrupted input more closely even when the model is closer to the clean SVG.

`misfit` and `pick` need only numpy; `transcribe` needs PyTorch (the `torch` extra).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from games.copycat import Svg, background, polish

if TYPE_CHECKING:
    from svgformer.model import SvgFormer

PICK_MARGIN = 2.0  # tuned on synthetic/validate: 1.96-2.11 on either half, either seed


def _pixels(rgba: bytes, width: int, height: int) -> np.ndarray:
    return np.frombuffer(rgba, np.uint8).reshape(height, width, 4).astype(np.float64) / 255


def misfit(svg: str, rgba: bytes, width: int, height: int, under: bytes | None = None) -> float:
    """Mean squared error (0..1 per channel) of `svg` drawn over the fitted background (`under`, or `background()` of
    the image) against the image, straight RGBA8."""
    under_px = _pixels(under if under is not None else background(rgba, width, height), width, height)[..., :3]
    top = _pixels(Svg(svg).render_rgba8(), width, height)
    alpha = top[..., 3:]
    drawn = top[..., :3] * alpha + under_px * (1 - alpha)
    return float(((drawn - _pixels(rgba, width, height)[..., :3]) ** 2).mean())


def pick(model_svg: str, tracer_svg: str, rgba: bytes, width: int, height: int, margin: float = PICK_MARGIN) -> str:
    """Which to use, "model" or "tracer": the tracer only if it fits the image `margin` times better."""
    under = background(rgba, width, height)
    model = misfit(model_svg, rgba, width, height, under)
    tracer = misfit(tracer_svg, rgba, width, height, under)
    return "tracer" if tracer * margin < model else "model"


def transcribe(
    model: SvgFormer,
    images: list[tuple[bytes, int, int]],
    device: str,
    precision: str = "bf16",
    batch: int = 64,
    polished: bool = True,
) -> list[tuple[str, float]]:
    """Each image's (straight RGBA8, width, height) SVG and seconds: its share of its batch's decoding, plus its polish.
    Every image must be the model's size."""
    import torch

    from svgformer import tokens as T
    from svgformer.decode import generate, to_images

    size = model.config.size
    if any((w, h) != (size, size) for _, w, h in images):
        raise ValueError(f"the model reads {size}x{size} images")
    autocast = device.startswith("cuda") and precision == "bf16"
    out: list[tuple[str, float]] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        rgb = np.stack(
            [np.frombuffer(r, np.uint8).reshape(size, size, 4)[..., :3].transpose(2, 0, 1) for r, _, _ in chunk]
        )
        began = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            sequences = generate(model, to_images(np.ascontiguousarray(rgb), device))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        share = (time.perf_counter() - began) / len(chunk)
        for (rgba, w, h), sequence in zip(chunk, sequences, strict=True):
            began = time.perf_counter()
            svg = T.decode(sequence, w)
            if polished:
                try:
                    svg = polish(Svg(svg), rgba, w, h).canonical()
                except ValueError:  # a translucent shape: polish composites opaque ones only
                    pass
            out.append((svg, share + time.perf_counter() - began))
    return out
