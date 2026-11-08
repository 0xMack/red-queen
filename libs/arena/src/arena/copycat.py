"""Copy Cat's evaluation protocol (docs/design/0022 Decisions 2-3).

Protocol `copycat.clean.v2`: every image of three held-out sets -- the checked-in `synthetic` set (`arena.copycat_sets`)
and the `icons` and `broad` artwork sets (`arena.copycat_artwork`, downloaded and built on first use), each corrupted by
`corrupt.v1` -- goes to the entrant, an `image -> SVG` function; its SVG is read in the subset and rendered by the game
core, and scored against the *clean* SVG the image was made from -- never against the corrupted input. An output outside
the subset (or the wrong size) is refused: scored as an empty, transparent SVG, and counted.

The columns, per image:

- **clean fidelity** (the rank): PSNR of the output's render against the clean render, composited over black and over
  white, averaged. A leftover background, a halo or a hole loses, and so does reproducing the noise. Capped at
  `PSNR_CAP` dB (identical renders);
- **alpha IoU**: intersection over union of the two renders' coverage (sum of min over sum of max);
- **layer recovery**: each clean part's best whole-shape IoU with any output shape, each rendered alone (the core's
  `layer_recovery`), averaged over parts;
- **structure**: shapes, path nodes and palette size, each as a ratio to the clean SVG's;
- **smoothness**: total absolute turning angle along the flattened outlines, as a ratio to the clean SVG's;
- **size**: the canonical serialization's bytes (time and memory are the evaluator's to measure).

An entrant's summary, per set, is the mean of each column, except the ratios, which are geometric means (a 10x and a
0.1x average to 1x, not 5x), each ratio floored at `RATIO_FLOOR`. Its overall summary is the same over the sets' summaries,
so each set counts the same, however many images it has (120 `broad` images would otherwise outweigh 24 `synthetic`
ones). Changing any of that is a new protocol version.

`copycat.clean.v1` was the same on the `synthetic` set alone.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from games.copycat import Svg, layer_recovery
from PIL import Image

from arena.copycat_sets import CORRUPTION, DATA, SEVERITIES, SPLITS

GAME = "copycat"
# each protocol's held-out sets; the last is the latest, which the page ranks and the targets routes serve
PROTOCOLS: dict[str, tuple[str, ...]] = {
    "copycat.clean.v1": ("synthetic",),
    "copycat.clean.v2": ("synthetic", "icons", "broad"),
}
PROTOCOL = "copycat.clean.v2"
SET = "synthetic"
SPLIT = "heldout"
# An entrant reads 8-bit RGB pixels and writes an SVG in the subset (docs/design/0007's interface, observer + output).
INTERFACE = "copycat/rgb8.v1+svg-subset.v1"
HELD_OUT_SEEDS: tuple[int, ...] = tuple(SPLITS[SPLIT])  # the checked-in held-out images, 6 per severity
PSNR_CAP = 60.0
RATIO_FLOOR = 0.01
RATIOS = ("shape_ratio", "node_ratio", "palette_ratio", "smoothness_ratio")
COLUMNS = ("fidelity", "psnr_black", "psnr_white", "alpha_iou", "layer_recovery", *RATIOS, "bytes")


@dataclass(frozen=True)
class Target:
    id: str
    seed: int
    severity: str
    levels: dict[str, Any]
    clean: str
    png: bytes
    width: int
    height: int
    icon: str | None = None  # an artwork set's image: the file its clean SVG came from (`copycat_artwork.family_of`)

    @property
    def set_name(self) -> str:
        return self.id.split("/")[0]

    def rgba8(self) -> bytes:
        return Image.open(io.BytesIO(self.png)).convert("RGBA").tobytes()


def load_set(split: str = SPLIT, root: Path | None = None, set_name: str = SET) -> list[Target]:
    """A split of a set (`synthetic` unless `set_name` says otherwise), in seed order: the checked-in held-out images
    unless `root` says otherwise (for `synthetic`, those in `HELD_OUT_SEEDS`)."""
    held_out = root is None and split == SPLIT and set_name == SET
    root = root or DATA / set_name / split
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    size = manifest["size"]
    return [
        Target(
            id=image["id"],
            seed=image["seed"],
            severity=image["levels"]["severity"],
            levels=image["levels"],
            clean=(root / f"{image['seed']}.svg").read_text(encoding="utf-8"),
            png=(root / f"{image['seed']}.png").read_bytes(),
            width=size,
            height=size,
            icon=image.get("icon"),
        )
        for image in manifest["images"]
        if image["corruption"] == CORRUPTION and (not held_out or image["seed"] in HELD_OUT_SEEDS)
    ]


def held_out(protocol: str = PROTOCOL, root: Path | None = None, download: bool = True) -> list[Target]:
    """Every image `protocol` scores, set by set, each in seed order. An artwork set is built under `root` (and downloaded
    first if `download` says so; `arena.copycat_sets.group_targets`)."""
    from arena.copycat_sets import group_targets

    sets = PROTOCOLS[protocol]
    if root is None and sets != (SET,):
        raise ValueError(f"{protocol} scores artwork sets, which are built under a root: pass one")
    return [t for set_name in sets for t in group_targets(f"{set_name}/{SPLIT}", root or DATA, download)]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return PSNR_CAP if mse <= 10 ** (-PSNR_CAP / 10) else min(PSNR_CAP, -10 * math.log10(mse))


def _premultiplied(svg: Svg) -> np.ndarray:
    return np.frombuffer(svg.render(), dtype=np.float64).reshape(svg.height, svg.width, 4)


def _ratio(output: float, clean: float) -> float:
    return max(RATIO_FLOOR, output / clean) if clean > 0 else 1.0


def score(target: Target, output: str) -> dict[str, Any]:
    """Every column for one image. Values are rounded (4 decimals) so a score reads the same on every platform."""
    clean = Svg(target.clean)
    refused = None
    try:
        svg = Svg(output)
        if (svg.width, svg.height) != (target.width, target.height):
            raise ValueError(f"the SVG is {svg.width}x{svg.height}, the image {target.width}x{target.height}")
    except ValueError as e:
        refused = str(e)
        svg = Svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {target.width} {target.height}"/>')
    ours, theirs = _premultiplied(svg), _premultiplied(clean)
    black = _psnr(ours[..., :3], theirs[..., :3])
    white = _psnr(ours[..., :3] + 1 - ours[..., 3:], theirs[..., :3] + 1 - theirs[..., 3:])
    a, b = ours[..., 3], theirs[..., 3]
    union = float(np.maximum(a, b).sum())
    recovered = layer_recovery(clean, svg)
    columns = {
        "fidelity": (black + white) / 2,
        "psnr_black": black,
        "psnr_white": white,
        "alpha_iou": float(np.minimum(a, b).sum()) / union if union > 0 else 1.0,
        "layer_recovery": sum(recovered) / len(recovered),
        "shape_ratio": _ratio(svg.shapes, clean.shapes),
        "node_ratio": _ratio(svg.nodes, clean.nodes),
        "palette_ratio": _ratio(svg.palette, clean.palette),
        "smoothness_ratio": _ratio(svg.turning, clean.turning),
    }
    return {
        "id": target.id,
        "severity": target.severity,
        **{k: round(v, 4) for k, v in columns.items()},
        "shapes": svg.shapes,
        "nodes": svg.nodes,
        "palette": svg.palette,
        "bytes": len(svg.canonical().encode()),
        "refused": refused,
    }


def summarize(images: list[dict[str, Any]]) -> dict[str, float]:
    """Each column over `images`: a mean, or a geometric mean for the ratios."""
    out: dict[str, float] = {}
    for column in COLUMNS:
        values = [image[column] for image in images]
        if column in RATIOS:
            out[column] = math.exp(sum(math.log(v) for v in values) / len(values))
        else:
            out[column] = sum(values) / len(values)
    out = {k: round(v, 4) for k, v in out.items()}
    out["refused"] = sum(image["refused"] is not None for image in images)
    return out


def summarize_sets(images: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """The overall summary and each set's (`summarize`): the overall one is each column's mean over the sets' summaries
    (geometric for the ratios), so every set weighs the same. With one set, it is that set's."""
    sets: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        sets.setdefault(image["id"].split("/")[0], []).append(image)
    by_set = {name: summarize(group) for name, group in sets.items()}
    if len(by_set) == 1:
        return next(iter(by_set.values())), by_set
    out: dict[str, float] = {}
    for column in COLUMNS:
        values = [summary[column] for summary in by_set.values()]
        if column in RATIOS:
            out[column] = round(math.exp(sum(math.log(v) for v in values) / len(values)), 4)
        else:
            out[column] = round(sum(values) / len(values), 4)
    out["refused"] = sum(summary["refused"] for summary in by_set.values())
    return out, by_set


def by_severity(images: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        severity: summarize(group)
        for severity in SEVERITIES
        if (group := [image for image in images if image["severity"] == severity])
    }


def compare(before: dict[str, Any], after: dict[str, Any], moved: float = 0.05) -> dict[str, Any]:
    """Two `copycat.report`s' groups: for each group in both, the means and the images that moved by more than `moved`
    dB, largest first."""
    groups = {}
    for group, now in after.items():
        if group not in before:
            continue
        was = before[group]["per_image"]
        changes = {seed: round(dB - was[seed], 3) for seed, dB in now["per_image"].items() if seed in was}
        groups[group] = {
            "before": before[group]["mean"],
            "after": now["mean"],
            "up": sum(change > moved for change in changes.values()),
            "down": sum(change < -moved for change in changes.values()),
            "moved": dict(sorted(((s, c) for s, c in changes.items() if abs(c) > moved), key=lambda sc: -abs(sc[1]))),
        }
    return groups
