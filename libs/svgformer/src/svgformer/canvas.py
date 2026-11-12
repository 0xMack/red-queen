"""svgformer that sees its canvas (docs/design/0025 Decision 2): the SVG is written one shape at a time, and before each
shape the encoder reads the target, the canvas drawn so far and their difference.

**The input** is ten channels of the image's size, uint8 like an image (`inputs`):
- the target (3);
- the canvas drawn over the background the tracer fits to the target (`games.copycat.background`) (3);
- the canvas's alpha (1): where something has been drawn at all;
- the residual, target minus canvas, as (difference + 1) / 2 (3).

**The training stream** (`prefix_sample`) is `svgformer.data`'s, one shape a sample: a random prefix of the clean SVG's
shapes in paint order (none to all of them) is decoded from its tokens and rendered as the canvas, and the target is the
next shape's tokens, `BOS shape END`, or `BOS END` once the prefix draws everything. Still a pure function of
`(stream seed, i)`.

**Decoding** (`draw`) is shape by shape: candidates for the next shape (greedy, plus samples), each rendered onto the
canvas; the one that most reduces the misfit against the input is kept, and with `verify` one that reduces nothing is no
shape. Drawing stops when the greedy candidate is `END`, when nothing improves, or at `max_shapes`.

The stream needs only numpy; `draw` needs PyTorch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from arena.copycat_sets import SEVERITIES, SIZE, corrupt_pixels
from games.copycat import Svg, background

from svgformer import tokens as T
from svgformer.data import ArtworkPool, mixed_draw

if TYPE_CHECKING:
    import torch

    from svgformer.model import SvgFormer

CHANNELS = 10
EMPTY = T.decode([T.BOS, T.END], SIZE)


@dataclass(frozen=True)
class PrefixSample:
    image: np.ndarray  # (CHANNELS, SIZE, SIZE) uint8: `inputs`
    tokens: list[int]  # BOS shape END, or BOS END
    seed: int
    severity: str
    shapes: int  # in the prefix
    total: int  # in the whole SVG


def rgba_array(rgba: bytes, width: int, height: int) -> np.ndarray:
    return np.frombuffer(rgba, np.uint8).reshape(height, width, 4)


def inputs(target_rgb: np.ndarray, under_rgb: np.ndarray, canvas_rgba: np.ndarray) -> np.ndarray:
    """(H, W, 3), (H, W, 3) and (H, W, 4) uint8 -> the encoder's (CHANNELS, H, W) uint8."""
    alpha = canvas_rgba[..., 3:].astype(np.float32) / 255
    drawn = canvas_rgba[..., :3].astype(np.float32) * alpha + under_rgb.astype(np.float32) * (1 - alpha)
    residual = (target_rgb.astype(np.float32) - drawn + 255) / 2
    out = np.concatenate([target_rgb.astype(np.float32), drawn, alpha * 255, residual], axis=-1)  # (H, W, CHANNELS)
    return np.rint(out).clip(0, 255).astype(np.uint8).transpose(2, 0, 1).copy()


def prefix_svg(shapes: list[list[int]]) -> str:
    return T.decode([T.BOS, *(t for s in shapes for t in s), T.END], SIZE)


def prefix_sample(
    stream_seed: int,
    i: int,
    severities: tuple[str, ...] = SEVERITIES,
    pool: ArtworkPool | None = None,
    artwork_share: float = 0.0,
) -> PrefixSample:
    return prefix_samples(stream_seed, i, severities, pool, artwork_share)[0]


def prefix_samples(
    stream_seed: int,
    i: int,
    severities: tuple[str, ...] = SEVERITIES,
    pool: ArtworkPool | None = None,
    artwork_share: float = 0.0,
    prefixes: int = 1,
) -> list[PrefixSample]:
    """`prefixes` samples of one corrupted image, at prefix lengths drawn without replacement (all of them, if the SVG
    has fewer). Corrupting the image and fitting its background are most of a sample's cost; a prefix's render is
    under a tenth of it. The first is `prefix_sample`'s."""
    svg, seed, severity = mixed_draw(stream_seed, i, severities, pool, artwork_share)
    pixels, _ = corrupt_pixels(svg, seed, severity)
    rgba = pixels.convert("RGBA")
    target = np.asarray(rgba)[..., :3]
    under = rgba_array(background(rgba.tobytes(), SIZE, SIZE), SIZE, SIZE)[..., :3]
    shapes = T._shapes(T.encode(svg, SIZE).tokens)
    rng = random.Random(f"svgformer-prefix:{stream_seed}:{i}")
    ks = [rng.randint(0, len(shapes))]
    rest = [k for k in range(len(shapes) + 1) if k != ks[0]]
    ks += rng.sample(rest, min(prefixes - 1, len(rest)))
    out = []
    for k in ks:
        canvas = rgba_array(Svg(prefix_svg(shapes[:k])).render_rgba8(), SIZE, SIZE)
        tokens = [T.BOS, *shapes[k], T.END] if k < len(shapes) else [T.BOS, T.END]
        out.append(PrefixSample(inputs(target, under, canvas), tokens, seed, severity, k, len(shapes)))
    return out


def _misfit(canvas_rgba: np.ndarray, target: np.ndarray, under: np.ndarray) -> float:
    alpha = canvas_rgba[..., 3:] / 255
    drawn = canvas_rgba[..., :3] / 255 * alpha + under * (1 - alpha)
    return float(((drawn - target) ** 2).mean())


@dataclass
class Drawing:
    """One image's drawing so far."""

    target: np.ndarray  # (H, W, 3) uint8
    under: np.ndarray  # (H, W, 3) uint8
    shapes: list[list[int]]
    canvas: np.ndarray  # (H, W, 4) uint8
    misfit: float
    base: str = EMPTY  # what the drawing started from: nothing, or another entrant's SVG (the tracer's)
    done: bool = False
    rejected: int = 0  # candidates' steps where nothing improved

    def input(self) -> np.ndarray:
        return inputs(self.target, self.under, self.canvas)

    def svg(self, shapes: list[list[int]] | None = None) -> str:
        """The base with `shapes` (by default the drawing's own) drawn over it."""
        return joined(self.base, prefix_svg(self.shapes if shapes is None else shapes))


def joined(under: str, over: str) -> str:
    """Two SVGs of one size as one, `over`'s elements painted after `under`'s."""
    a, b = Svg(under).canonical().strip().splitlines(), Svg(over).canonical().strip().splitlines()
    return Svg("\n".join(a[:-1] + b[1:])).canonical()


def start(rgba: bytes, width: int, height: int, base: str | None = None) -> Drawing:
    if (width, height) != (SIZE, SIZE):
        raise ValueError(f"svgformer draws {SIZE} px images, not {width}x{height}")
    px = rgba_array(rgba, width, height)
    target, under = px[..., :3], rgba_array(background(rgba, width, height), width, height)[..., :3]
    base = Svg(base).canonical() if base is not None else EMPTY
    canvas = rgba_array(Svg(base).render_rgba8(), SIZE, SIZE)
    return Drawing(target.copy(), under.copy(), [], canvas, _misfit(canvas, target / 255, under / 255), base)


def draw(
    model: SvgFormer,
    images: list[tuple[bytes, int, int]],
    device: torch.device | str,
    samples: int = 0,
    temperature: float = 0.7,
    verify: bool = True,
    max_shapes: int = 48,
    seed: int = 0,
    precision: str = "bf16",
    bases: list[str] | None = None,
    max_shape_tokens: int = 512,
) -> list[str]:
    """Each image's (straight RGBA8, width, height) SVG, drawn shape by shape: per step the greedy candidate plus
    `samples` sampled at `temperature`, each at most `max_shape_tokens` (a longer one is a loop, and dropped). With
    `bases`, each drawing starts from that SVG (0025 R2: the model adds what the tracer missed)."""
    import torch

    from svgformer.decode import generate

    drawings = [start(*image, None if bases is None else bases[k]) for k, image in enumerate(images)]
    generator = torch.Generator(device=device).manual_seed(seed)
    autocast = str(device).startswith("cuda") and precision == "bf16"
    for _ in range(max_shapes):
        active = [d for d in drawings if not d.done]
        if not active:
            break
        batch = torch.from_numpy(np.stack([d.input() for d in active])).to(device).float() / 255
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            greedy = generate(model, batch, max_tokens=max_shape_tokens, one_shape=True)
            sampled = (
                generate(
                    model,
                    batch.repeat_interleave(samples, dim=0),
                    max_tokens=max_shape_tokens,
                    one_shape=True,
                    temperature=temperature,
                    generator=generator,
                )
                if samples
                else []
            )
        for j, d in enumerate(active):
            if greedy[j][1] == T.END:
                d.done = True
                continue
            target, under = d.target / 255, d.under / 255
            best = None
            for candidate in [greedy[j], *sampled[j * samples : (j + 1) * samples]]:
                shape = T._shapes(candidate)
                if not shape:
                    continue  # END, or a shape cut off at the length cap
                canvas = rgba_array(Svg(d.svg([*d.shapes, shape[0]])).render_rgba8(), SIZE, SIZE)
                misfit = _misfit(canvas, target, under)
                if best is None or misfit < best[0]:
                    best = (misfit, shape[0], canvas)
            if best is None or (verify and best[0] >= d.misfit):
                d.rejected += 1
                d.done = True
                continue
            d.misfit, d.canvas = best[0], best[2]
            d.shapes.append(best[1])
    return [d.svg() for d in drawings]
