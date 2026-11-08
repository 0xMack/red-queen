"""Copy Cat leaderboard evaluation (docs/design/0022): every entrant turns each held-out image into an SVG, scored
against the clean SVG behind it under protocol `copycat.clean.v2` (`arena.copycat`: the held-out `synthetic`, `icons`
and `broad` sets; `copycat.clean.v1` was `synthetic` alone, and still runs).

Phase 1's entrants are the baselines of the ladder's first rung, all of them tracing or approximating the *input*:

- **VTracer** (the `vtracer` package, visioncortex's Rust tracer) at three presets: its defaults, a coarse one for flat
  art (fewer colours, bigger speckles filtered), and its polygon mode. Its SVG positions every path with a
  `transform="translate(x,y)"`, which the subset has no room for, so the adapter here applies the translation to the
  path's coordinates -- the same shapes, written in the subset. Nothing else is changed.
- **A `primitive`-style hill climb** (the game core's `copycat_climb.rs`) at two shape budgets.
- **Each family with its bottom layer deleted**, the background a person would delete by hand: none of them makes
  anything transparent, so without this every entrant loses the same ~20 dB over black, and the comparison says nothing
  else.

Phase 2's entrant, the ladder's second rung, recovers the clean SVG instead:

- **The MDL tracer** (the game core's `copycat_trace.rs`, entrant `baseline:structured`, versioned): noise and background estimated, an OKLab palette
  merged by description length, cleaned labels, parts ordered and each fitted with its shortest shape -- a transparent,
  layered SVG. Untrained (its settings were tuned on the `train` split only), so it is a baseline in the leaderboard's
  sense, like every entrant so far.

Each entrant runs in a fresh process, so its time per image and its peak memory (above that process's own at start)
are its alone; scoring happens here, after.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import re
import statistics
import time
from typing import Any

from arena.copycat import (
    GAME,
    INTERFACE,
    PROTOCOLS,
    SPLIT,
    Target,
    by_severity,
    held_out,
    score,
    summarize_sets,
)
from arena.copycat_sets import CORRUPTION
from arena.costs import hardware_fingerprint, peak_rss_bytes
from jobcore import Sink, experiments_dir
from telemetry import EvaluationRecord

from evaluator.protocols.copycat_report import commit
from evaluator.registry import protocol

# The MDL tracer's version: one more for every change kept, each listed in apps/frontend/app/data/copycatVersions.ts
# (which the Learn case study draws). Its entrant id stays `baseline:structured`, so the leaderboard holds only the
# current version; the record notes the version and the commit it was scored at.
MDL_TRACER_VERSION = 23

# key -> (label, description, how it runs: the tracer and its settings)
VTRACER_DEFAULTS: dict[str, Any] = {"colormode": "color", "hierarchical": "stacked", "mode": "spline"}
ENTRANTS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "vtracer": (
        "VTracer (defaults)",
        "VTracer's colour tracer with its default settings: stacked layers, spline curves, speckles under 4 px filtered.",
        {"tracer": "vtracer", **VTRACER_DEFAULTS},
    ),
    "vtracer-flat": (
        "VTracer (flat art)",
        (
            "VTracer set for flat-colour art: coarser colour precision, larger gradient steps between layers, "
            "speckles under 16 px filtered."
        ),
        {"tracer": "vtracer", **VTRACER_DEFAULTS, "filter_speckle": 16, "color_precision": 4, "layer_difference": 32},
    ),
    "vtracer-polygon": (
        "VTracer (polygons)",
        "VTracer's defaults with straight-line outlines instead of splines.",
        {"tracer": "vtracer", **VTRACER_DEFAULTS, "mode": "polygon"},
    ),
    "vtracer-nobg": (
        "VTracer (defaults), background deleted",
        "VTracer with its default settings, then its bottom layer -- the background -- deleted.",
        {"tracer": "vtracer", **VTRACER_DEFAULTS, "drop_bottom": True},
    ),
    "hillclimb-16": (
        "Hill climb (16 shapes)",
        (
            "A primitive-style hill climb: a background rect, then 16 opaque triangles, rects, ellipses or circles, "
            "each the best of 100 random shapes climbed until 100 mutations in a row fail."
        ),
        {"tracer": "hillclimb", "shapes": 16, "candidates": 100, "patience": 100},
    ),
    "hillclimb-64": (
        "Hill climb (64 shapes)",
        "The same hill climb with 64 shapes.",
        {"tracer": "hillclimb", "shapes": 64, "candidates": 100, "patience": 100},
    ),
    "hillclimb-64-nobg": (
        "Hill climb (64 shapes), background deleted",
        "The 64-shape hill climb, then its first shape -- the background rect -- deleted.",
        {"tracer": "hillclimb", "shapes": 64, "candidates": 100, "patience": 100, "drop_bottom": True},
    ),
    "structured": (
        f"MDL tracer {MDL_TRACER_VERSION}",
        (
            "Copy Cat's minimum-description-length tracer: noise and background estimated, an OKLab palette merged by "
            "description length, cleaned labels, parts ordered by how much of each other they cover, holes cut, and "
            "each part fitted with its shortest shape (circle, ellipse, rect, rounded rect, polygon, smooth spline or "
            "Bezier path), hidden parts completed; then every shape polished against the whole image, adding a point "
            "only where the error it saves pays for it. Transparent, one group per shape."
        ),
        {"tracer": "structured"},
    ),
}

_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PATH = re.compile(r'<path d="([^"]*)" fill="([^"]*)" transform="translate\(([^,]+),([^)]+)\)"\s*/>')


def vtracer_to_subset(svg: str) -> str:
    """VTracer's SVG with each path's `translate` applied to its (absolute, x/y-paired) coordinates."""
    paths = []
    for d, fill, tx, ty in _PATH.findall(svg):
        offsets = (float(tx), float(ty))
        pieces = _NUMBER.split(d)  # text between numbers, then each number, alternating
        numbers = _NUMBER.findall(d)
        moved = [repr(float(n) + offsets[i % 2]) for i, n in enumerate(numbers)]
        text = "".join(p + m for p, m in zip(pieces, [*moved, ""], strict=True))
        paths.append(f'<path d="{text}" fill="{fill}"/>')
    size = re.search(r'<svg[^>]*width="(\d+)" height="(\d+)"', svg)
    width, height = size.groups() if size else ("0", "0")
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n' + "\n".join(paths) + "\n</svg>"


def drop_bottom(svg: str) -> str:
    """The SVG without its first shape (in canonical form)."""
    from games.copycat import Svg

    lines = Svg(svg).canonical().splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if i > 0 and not line.startswith(("<g", "</g")))
    return "".join(lines[:first] + lines[first + 1 :])


def trace(settings: dict[str, Any], target: Target) -> str:
    """One entrant on one image: its SVG."""
    settings = dict(settings)
    tracer = settings.pop("tracer")
    bottom_dropped = settings.pop("drop_bottom", False)
    if tracer == "vtracer":
        import vtracer

        svg = vtracer_to_subset(vtracer.convert_raw_image_to_svg(target.png, img_format="png", **settings))
    elif tracer == "hillclimb":
        from games.copycat import hill_climb

        svg = hill_climb(target.rgba8(), target.width, target.height, seed=target.seed, **settings).canonical()
    elif tracer == "structured":
        from games.copycat import trace as structured

        svg = structured(target.rgba8(), target.width, target.height, **settings).canonical()
    else:
        raise ValueError(f"unknown tracer {tracer}")
    return drop_bottom(svg) if bottom_dropped else svg


def run_entrant(settings: dict[str, Any], targets: list[Target]) -> dict[str, Any]:
    """In the entrant's own process: every image's SVG and time, and the process's memory before and at peak."""
    baseline = peak_rss_bytes()
    svgs, image_us = [], []
    for target in targets:
        start = time.perf_counter()
        svgs.append(trace(settings, target))
        image_us.append((time.perf_counter() - start) * 1e6)
    return {"svgs": svgs, "image_us": image_us, "baseline_rss_bytes": baseline, "peak_rss_bytes": peak_rss_bytes()}


def ci95(images: list[dict[str, Any]]) -> float:
    """The 95% interval's half-width of the mean of the sets' mean fidelities (one set: of the plain mean)."""
    sets: dict[str, list[float]] = {}
    for image in images:
        sets.setdefault(image["id"].split("/")[0], []).append(image["fidelity"])
    if len(sets) == 1:
        (fidelity,) = sets.values()
        sd = statistics.stdev(fidelity) if len(fidelity) > 1 else 0.0
        return round(1.96 * sd / len(fidelity) ** 0.5, 4)
    variance = sum(statistics.variance(f) / len(f) if len(f) > 1 else 0.0 for f in sets.values()) / len(sets) ** 2
    return round(1.96 * variance**0.5, 4)


def record_for(protocol_name: str, key: str, targets: list[Target], hardware: dict[str, Any]) -> EvaluationRecord:
    label, description, settings = ENTRANTS[key]
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        ran = pool.submit(run_entrant, settings, targets).result()
    images = [
        {**score(target, svg), "image_us": us, "svg": svg}
        for target, svg, us in zip(targets, ran["svgs"], ran["image_us"], strict=True)
    ]
    summary, by_set = summarize_sets(images)
    fidelity = [image["fidelity"] for image in images]
    sets = PROTOCOLS[protocol_name]
    peak, baseline = ran["peak_rss_bytes"], ran["baseline_rss_bytes"]
    return EvaluationRecord(
        game=GAME,
        protocol=protocol_name,
        entrant_id=f"baseline:{key}",
        entrant_kind="baseline",
        label=label,
        interface=INTERFACE,
        created_at=time.time(),
        metrics={
            "quality": {
                "n": len(fidelity),
                "mean": summary["fidelity"],
                "ci95": ci95(images),
                "median": statistics.median(fidelity),
                "min": min(fidelity),
                "max": max(fidelity),
                "scores": fidelity,
            },
            "inference": {
                "total_us": statistics.fmean(ran["image_us"]),
                "parameters": 0,
                "artifact_bytes": round(summary["bytes"]),
            },
            "training": {"measured": True, "none": True},
            "model": {"description": description, "observer_level": 1, "note": None, "shape": {"algorithm": label}},
            "protocol": {
                "set": "+".join(sets),
                "split": SPLIT,
                "corruption": CORRUPTION,
                "images": len(targets),
                **(
                    {"held_out_seeds": [targets[0].seed, targets[-1].seed]}
                    if len(sets) == 1
                    else {"images_per_set": {name: sum(t.set_name == name for t in targets) for name in sets}}
                ),
                "metric": (
                    "clean fidelity: PSNR (dB) against the clean render over black and over white, averaged"
                    + ("" if len(sets) == 1 else "; the mean of each set's mean")
                ),
            },
            "copycat": {
                "settings": settings,
                "commit": commit(),
                **({"version": MDL_TRACER_VERSION} if key == "structured" else {}),
                "columns": summary,
                **({"by_set": by_set} if len(sets) > 1 else {}),
                "by_severity": by_severity(images),
                "images": images,
                "memory": {
                    "peak_rss_bytes": peak,
                    "baseline_rss_bytes": baseline,
                    "peak_above_baseline_bytes": peak - baseline if peak and baseline else None,
                },
            },
        },
        hardware={**hardware, "engine": "rust"},
    )


def table(records: list[EvaluationRecord]) -> str:
    columns = ("fidelity", "alpha_iou", "layer_recovery", "shape_ratio", "palette_ratio", "smoothness_ratio")
    lines = [
        "| entrant | group | " + " | ".join(columns) + " |",
        "|---|---|" + "---|" * len(columns),
    ]
    for record in records:
        cc = record.metrics["copycat"]
        rows = [("all", cc["columns"]), *cc.get("by_set", {}).items(), *cc["by_severity"].items()]
        for group, row in rows:
            lines.append(f"| {record.label} | {group} | " + " | ".join(f"{row[c]:.3g}" for c in columns) + " |")
    return "\n".join(lines)


def run(protocol_name: str, sink: Sink) -> dict[str, Any]:
    store = sink.evaluations
    hardware = hardware_fingerprint()
    targets = held_out(protocol_name, experiments_dir() / "copycat-sets")
    records = [record_for(protocol_name, key, targets, hardware) for key in ENTRANTS]
    for record in records:
        store.put(record)
    removed = store.prune(GAME, protocol_name, {r.entrant_id for r in records})
    for entrant_id in removed:
        print(f"removed {entrant_id}: no longer an entrant")
    records.sort(key=lambda r: -r.metrics["quality"]["mean"])
    sets = "+".join(PROTOCOLS[protocol_name])
    print(f"Clean fidelity (dB) and the other columns on {len(targets)} held-out {sets} images ({CORRUPTION})\n")
    print(table(records))
    return {"protocol": protocol_name, "evaluated": sorted(r.entrant_id for r in records), "removed": sorted(removed)}


for _name in PROTOCOLS:
    protocol(_name)(lambda spec, params, sink, _name=_name: run(_name, sink))
