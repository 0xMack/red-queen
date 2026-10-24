"""Copy Cat's target sets and its corruption model (docs/design/0022 Decision 1): clean SVGs with every part known whole,
corrupted the way an image generator would corrupt them.

- **`synthetic`**: from a seed, a stack of 2-8 whole parts -- rects, circles, ellipses, polygons and smooth Bézier
  blobs -- in 2-6 palette colours on a transparent 128 px canvas. Each part stays at least a quarter visible, so it can
  be recovered, and parts that overlap get different colours where the palette allows.
- **`corrupt.v1`**: a seeded, versioned corruption at one of four severities (`clean` is the control: the bare render on
  a flat background). In order: per-part colour drift, a faint shading gradient, a slight warp, blur, a background
  (near-flat, a soft gradient, or clutter), Gaussian noise and film grain, then JPEG at quality 50-90 (Pillow). Every
  level drawn is recorded, so results can be broken down by corruption.

Like an observer, `corrupt.v1` is never edited in place: a change is `corrupt.v2`. Its output depends on Pillow's JPEG
encoder, which is why the held-out images are generated once and checked in beside their clean SVGs
(`data/copycat/synthetic/heldout/`, with `manifest.json`), rather than regenerated:

    uv run python -m arena.copycat_sets            # rewrite the checked-in held-out set
    uv run python -m arena.copycat_sets --split train --out DIR
"""

from __future__ import annotations

import argparse
import colorsys
import io
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from games.copycat import Svg
from PIL import Image

CORRUPTION = "corrupt.v1"
SIZE = 128
SEVERITIES = ("clean", "light", "medium", "heavy")
DATA = Path(__file__).parent / "data" / "copycat"

# The splits' seeds: training from 0, held out from 10,000 (the leaderboard scores only those). Severity cycles with
# the seed, so every severity has a quarter of each split.
SPLITS: dict[str, range] = {"train": range(96), "heldout": range(10_000, 10_024)}

# Each severity's range per level, drawn uniformly per image. Colour levels are in 8-bit steps, blur and warp in px.
LEVELS: dict[str, dict[str, tuple[float, float]]] = {
    "clean": {},
    "light": {
        "drift": (0, 4),
        "shading": (0, 0.04),
        "warp": (0, 0.5),
        "blur": (0.3, 0.6),
        "noise": (0, 2),
        "grain": (0, 2),
        "jpeg": (85, 90),
    },
    "medium": {
        "drift": (4, 10),
        "shading": (0.04, 0.1),
        "warp": (0.5, 1.0),
        "blur": (0.6, 1.0),
        "noise": (2, 5),
        "grain": (2, 4),
        "jpeg": (70, 85),
    },
    "heavy": {
        "drift": (10, 18),
        "shading": (0.1, 0.18),
        "warp": (1.0, 1.8),
        "blur": (1.0, 1.6),
        "noise": (5, 9),
        "grain": (4, 7),
        "jpeg": (50, 70),
    },
}
BACKGROUNDS: dict[str, tuple[str, ...]] = {
    "clean": ("flat",),
    "light": ("flat", "gradient"),
    "medium": ("flat", "gradient", "clutter"),
    "heavy": ("gradient", "clutter", "clutter"),
}


def severity_of(seed: int) -> str:
    return SEVERITIES[seed % len(SEVERITIES)]


# --- synthetic --------------------------------------------------------------------------------------------------


def _palette(rng: random.Random, k: int) -> list[tuple[int, int, int]]:
    """k colours far apart (in RGB), saturated or dark enough to read on a light background."""
    colours: list[tuple[int, int, int]] = []
    while len(colours) < k:
        h, s, v = rng.random(), rng.uniform(0.45, 0.95), rng.uniform(0.3, 0.9)
        c = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
        if all(math.dist(c, other) > 90 for other in colours):
            colours.append(c)
    return colours


def _blob(rng: random.Random, cx: float, cy: float, r: float) -> str:
    """A smooth closed path: k points around a centre at jittered radii, joined by Catmull-Rom cubics."""
    k = rng.randint(4, 7)
    turn = rng.uniform(0, 2 * math.pi)
    points = []
    for i in range(k):
        a = turn + 2 * math.pi * (i + rng.uniform(-0.2, 0.2)) / k
        radius = r * rng.uniform(0.6, 1.0)
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    d = [f"M{points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(k):
        p0, p1, p2, p3 = (points[(i + j) % k] for j in (-1, 0, 1, 2))
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}")
    return f'<path d="{"".join(d)}Z"'


def _part(rng: random.Random) -> tuple[str, tuple[float, float, float, float]]:
    """One part's element (without its fill) and its bounding box."""
    kind = rng.choice(["rect", "circle", "ellipse", "polygon", "blob", "blob"])
    cx, cy = rng.uniform(24, SIZE - 24), rng.uniform(24, SIZE - 24)
    r = rng.uniform(10, 34)
    if kind == "rect":
        w, h = r * rng.uniform(1.0, 2.0), r * rng.uniform(1.0, 2.0)
        x, y = cx - w / 2, cy - h / 2
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"', (x, y, x + w, y + h)
    if kind == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"', (cx - r, cy - r, cx + r, cy + r)
    if kind == "ellipse":
        rx, ry = r * rng.uniform(0.5, 1.0), r * rng.uniform(0.5, 1.0)
        return f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"', (cx - rx, cy - ry, cx + rx, cy + ry)
    if kind == "polygon":
        n = rng.randint(3, 6)
        turn = rng.uniform(0, 2 * math.pi)
        points = [
            (cx + r * math.cos(turn + 2 * math.pi * i / n), cy + r * math.sin(turn + 2 * math.pi * i / n))
            for i in range(n)
        ]
        text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return f'<polygon points="{text}"', (cx - r, cy - r, cx + r, cy + r)
    return _blob(rng, cx, cy, r), (cx - r, cy - r, cx + r, cy + r)


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _visible_fractions(svg: Svg) -> list[float]:
    """Each part's share of its own area left uncovered by the (opaque) parts above it."""
    masks = [np.frombuffer(svg.coverage(i), dtype=np.float64) for i in range(svg.shapes)]
    above = np.zeros(SIZE * SIZE)
    out = []
    for mask in reversed(masks):
        area = mask.sum()
        out.append(float((mask * (1 - above)).sum() / area) if area > 0 else 0.0)
        above = 1 - (1 - above) * (1 - mask)
    return out[::-1]


def synthetic(seed: int) -> tuple[str, dict[str, Any]]:
    """The clean SVG for `seed` (canonical) and what it is made of."""
    rng = random.Random(f"copycat.synthetic:{seed}")
    while True:
        k = rng.randint(2, 6)
        palette = _palette(rng, k)
        n = rng.randint(k, min(k + 3, 8))
        parts: list[tuple[str, tuple[float, ...]]] = [_part(rng) for _ in range(n)]
        fills: list[tuple[int, int, int]] = []
        order = rng.sample(palette, k)  # every colour used once, then reuse
        for i, (_element, box) in enumerate(parts):
            if i < k:
                fills.append(order[i])
                continue
            clash = {fills[j] for j in range(i) if _overlaps(parts[j][1], box)}
            fills.append(rng.choice([c for c in palette if c not in clash] or palette))
        body = "".join(f'{e} fill="#{c[0]:02x}{c[1]:02x}{c[2]:02x}"/>' for (e, _), c in zip(parts, fills, strict=True))
        svg = Svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">{body}</svg>')
        if min(_visible_fractions(svg)) >= 0.25:
            return svg.canonical(), {
                "parts": n,
                "palette": len(set(fills)),
                "kinds": [kind for kind, _fill, _opacity in svg.describe()],
            }


# --- corrupt.v1 -----------------------------------------------------------------------------------------------------


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian over the first two axes, edges extended."""
    if sigma <= 0:
        return image
    radius = max(1, math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    out = image
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in range(out.ndim)], mode="edge")
        out = sum(w * np.take(padded, np.arange(i, i + out.shape[axis]), axis=axis) for i, w in enumerate(kernel))
    return out


def _warp(image: np.ndarray, amplitude: float, rng: random.Random) -> np.ndarray:
    """A smooth sinusoidal displacement of up to `amplitude` px, bilinear resampling (transparent outside)."""
    if amplitude <= 0:
        return image
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    lx, ly = rng.uniform(40, 90), rng.uniform(40, 90)
    px, py = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    sx = xs + amplitude * np.sin(2 * math.pi * ys / ly + py)
    sy = ys + amplitude * np.sin(2 * math.pi * xs / lx + px)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))

    def at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return padded[np.clip(y, -1, SIZE) + 1, np.clip(x, -1, SIZE) + 1]

    top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
    bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
    return top * (1 - fy) + bottom * fy


def _background(kind: str, rng: random.Random) -> tuple[np.ndarray, dict[str, Any]]:
    """An opaque RGB background in 0..1 and how it was drawn."""
    base = np.array([rng.uniform(0.86, 1.0) for _ in range(3)])
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64) / (SIZE - 1)
    if kind == "flat":
        return np.broadcast_to(base, (SIZE, SIZE, 3)).copy(), {"kind": kind}
    angle = rng.uniform(0, 2 * math.pi)
    t = ((xs - 0.5) * math.cos(angle) + (ys - 0.5) * math.sin(angle) + 0.5)[..., None]
    other = np.clip(base + np.array([rng.uniform(-0.12, 0.06) for _ in range(3)]), 0, 1)
    image = base * (1 - t) + other * t
    if kind == "gradient":
        return image, {"kind": kind}
    # clutter: faint unrelated shapes and specks behind the subject -- what a generator leaves in an "empty" background
    count = rng.randint(4, 10)
    body = []
    for _ in range(count):
        element, _box = _part(rng)
        shade = [round(255 * min(1.0, c * rng.uniform(0.8, 1.05))) for c in base]
        body.append(
            f'{element} fill="#{shade[0]:02x}{shade[1]:02x}{shade[2]:02x}" opacity="{rng.uniform(0.3, 0.7):.2f}"/>'
        )
    for _ in range(rng.randint(10, 30)):
        x, y, r = rng.uniform(0, SIZE), rng.uniform(0, SIZE), rng.uniform(0.6, 1.8)
        grey = round(255 * rng.uniform(0.55, 0.85))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#{grey:02x}{grey:02x}{grey:02x}"/>')
    clutter = Svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">{"".join(body)}</svg>')
    layer = np.frombuffer(clutter.render(), dtype=np.float64).reshape(SIZE, SIZE, 4)
    image = layer[..., :3] + image * (1 - layer[..., 3:])
    return image, {"kind": kind, "shapes": count}


def corrupt(clean_svg: str, seed: int, severity: str) -> tuple[bytes, dict[str, Any]]:
    """`corrupt.v1` of a clean SVG: the image as 8-bit RGB PNG bytes, and every level drawn."""
    rng = random.Random(f"{CORRUPTION}:{seed}:{severity}")
    noise_rng = np.random.default_rng(rng.getrandbits(64))
    ranges = LEVELS[severity]
    levels: dict[str, Any] = {name: rng.uniform(lo, hi) for name, (lo, hi) in ranges.items()}
    # 1. per-part colour drift: each part's fill moved a little, independently per channel
    drift = levels.get("drift", 0.0)
    body = clean_svg
    if drift > 0:
        lines = []
        for line in clean_svg.splitlines():
            if 'fill="#' in line:
                start = line.index('fill="#') + 7
                rgb = [int(line[start + 2 * i : start + 2 * i + 2], 16) for i in range(3)]
                rgb = [min(255, max(0, round(c + rng.gauss(0, drift)))) for c in rgb]
                line = line[:start] + "".join(f"{c:02x}" for c in rgb) + line[start + 6 :]
            lines.append(line)
        body = "\n".join(lines)
    subject = np.frombuffer(Svg(body).render(), dtype=np.float64).reshape(SIZE, SIZE, 4).copy()
    # 2. a faint shading gradient across the subject
    shading = levels.get("shading", 0.0)
    if shading > 0:
        angle = rng.uniform(0, 2 * math.pi)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64) / (SIZE - 1)
        t = (xs - 0.5) * math.cos(angle) + (ys - 0.5) * math.sin(angle)
        subject[..., :3] *= (1 + 2 * shading * t)[..., None]
    # 3-4. a slight warp, then blur (soft anti-aliasing), on premultiplied colour
    subject = _warp(subject, levels.get("warp", 0.0), rng)
    subject = _gaussian_blur(subject, levels.get("blur", 0.0))
    # 5. the background the subject sits on
    kind = rng.choice(BACKGROUNDS[severity])
    background, levels["background"] = _background(kind, rng)
    image = subject[..., :3] + background * (1 - subject[..., 3:])
    # 6. Gaussian noise per channel, and film grain (blurred luminance noise)
    if levels.get("noise", 0.0) > 0:
        image = image + noise_rng.normal(0, levels["noise"] / 255, image.shape)
    if levels.get("grain", 0.0) > 0:
        grain = _gaussian_blur(noise_rng.normal(0, 1, (SIZE, SIZE)), 0.8)
        image = image + (levels["grain"] / 255 / grain.std()) * grain[..., None]
    pixels = Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), "RGB")
    # 7. JPEG, as a generated image usually arrives
    if "jpeg" in levels:
        levels["jpeg"] = round(levels["jpeg"])
        buffer = io.BytesIO()
        pixels.save(buffer, "JPEG", quality=levels["jpeg"])
        pixels = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    out = io.BytesIO()
    pixels.save(out, "PNG", optimize=True)
    rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in levels.items()}
    return out.getvalue(), {"severity": severity, **rounded}


def write_split(split: str, out: Path) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in SPLITS[split]:
        clean, parts = synthetic(seed)
        severity = severity_of(seed)
        png, levels = corrupt(clean, seed, severity)
        (out / f"{seed}.svg").write_text(clean, encoding="utf-8", newline="\n")
        (out / f"{seed}.png").write_bytes(png)
        manifest.append({"id": f"synthetic/{seed}", "seed": seed, "corruption": CORRUPTION, "levels": levels, **parts})
    text = json.dumps({"set": "synthetic", "split": split, "size": SIZE, "images": manifest}, indent=1) + "\n"
    (out / "manifest.json").write_text(text, encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLITS), default="heldout")
    parser.add_argument("--out", type=Path, help="default: the checked-in set (held-out only)")
    args = parser.parse_args()
    if args.out is None and args.split != "heldout":
        parser.error("only the held-out split is checked in: pass --out")
    out = args.out or DATA / "synthetic" / args.split
    images = write_split(args.split, out)
    print(f"wrote {len(images)} images to {out}")


if __name__ == "__main__":
    main()
