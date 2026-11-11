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

Phase P3 of docs/design/0024 adds the first trained entrants, from every completed `copycat.svgformer` run outside an
experiment:

- **`run:<id>`**: the run's transformer, greedy and grammar-constrained, then the MDL tracer's polish against the input
  (`svgformer.entrant.transcribe`).
- **`run:<id>+structured`**: the same, or the MDL tracer's SVG, per image (`svgformer.entrant.pick`): the tracer's only
  when it fits the input twice as well.

They need PyTorch (`svgformer[torch]`) and run on the GPU when there is one; without PyTorch they are skipped, loudly.

Each entrant runs in a fresh process, so its time per image and its peak memory (above that process's own at start)
are its alone; scoring happens here, after.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
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
from telemetry import EvaluationRecord, RunInfo

from evaluator.protocols.copycat_report import commit
from evaluator.registry import protocol

# The MDL tracer's version: one more for every change kept, each listed in apps/frontend/app/data/copycatVersions.ts
# (which the Learn case study draws). Its entrant id stays `baseline:structured`, so the leaderboard holds only the
# current version; the record notes the version and the commit it was scored at.
MDL_TRACER_VERSION = 27

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


def run_entrant(settings: dict[str, Any], targets: list[Target], weights: bytes | None = None) -> dict[str, Any]:
    """In the entrant's own process: every image's SVG and time, and the process's memory before and at peak."""
    if settings["tracer"] == "svgformer":
        return run_svgformer(settings, targets, weights)
    baseline = peak_rss_bytes()
    svgs, image_us = [], []
    for target in targets:
        start = time.perf_counter()
        svgs.append(trace(settings, target))
        image_us.append((time.perf_counter() - start) * 1e6)
    return {"svgs": svgs, "image_us": image_us, "baseline_rss_bytes": baseline, "peak_rss_bytes": peak_rss_bytes()}


def run_svgformer(settings: dict[str, Any], targets: list[Target], weights: bytes | None) -> dict[str, Any]:
    """`run_entrant` for a trained svgformer: batched on the GPU if there is one, so an image's time is its share of its
    batch's decoding plus its own polish (and, paired, the tracer and the pick)."""
    import torch
    from games.copycat import trace as structured
    from svgformer.entrant import pick, transcribe
    from svgformer.model import SvgFormer

    if weights is None:
        raise ValueError("an svgformer entrant needs its weights")
    baseline = peak_rss_bytes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SvgFormer.load(weights, device)
    images = [(t.rgba8(), t.width, t.height) for t in targets]
    out = transcribe(model, images, device, settings["precision"])
    svgs, image_us = [svg for svg, _ in out], [seconds * 1e6 for _, seconds in out]
    chosen = None
    if settings.get("paired_with"):
        chosen = []
        for i, (rgba, width, height) in enumerate(images):
            start = time.perf_counter()
            traced = structured(rgba, width, height).canonical()
            which = pick(svgs[i], traced, rgba, width, height, settings["margin"])
            image_us[i] += (time.perf_counter() - start) * 1e6
            if which == "tracer":
                svgs[i] = traced
            chosen.append(which)
    return {
        "svgs": svgs,
        "image_us": image_us,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak_rss_bytes(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "chosen": chosen,
    }


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


def svgformer_entrants(sink: Sink) -> list[dict[str, Any]]:
    """Two entrants per completed `copycat.svgformer` run outside an experiment (an experiment's runs are compared there,
    not ranked: docs/design/0018)."""
    if importlib.util.find_spec("torch") is None:
        print("skipping svgformer entrants: PyTorch isn't installed (uv sync --all-packages --extra torch)")
        return []
    from svgformer.entrant import PICK_MARGIN

    entrants = []
    for run in sink.registry.list_runs():
        config = run.config or {}
        if config.get("game") != GAME or config.get("representation") != "svgformer" or run.status != "completed":
            continue
        if config.get("experiment"):
            continue
        ref = f"{run.run_id}-weights"
        weights = sink.artifacts.get_program(ref)
        size, steps = config["training"]["model"], config["training"]["steps"]  # model.size is the image side
        name = f"SvgFormer {size} ({steps // 1000}k steps)"
        model = (
            f"a {config['parameters'] / 1e6:.1f}M-parameter encoder-decoder transformer trained on the synthetic "
            f"stream for {steps:,} steps; greedy, grammar-constrained decoding, then the MDL tracer's polish"
        )
        settings = {
            "tracer": "svgformer",
            "run": run.run_id,
            "precision": config["training"].get("precision", "bf16"),
            "decoding": "greedy",
            "polish": True,
        }
        common = {"run": run, "champion_ref": ref, "weights": weights, "parameters": config["parameters"]}
        entrants.append(
            {
                **common,
                "entrant_id": f"run:{run.run_id}",
                "label": name,
                "algorithm": f"SvgFormer {size}",
                "description": model[0].upper() + model[1:] + ".",
                "settings": settings,
            }
        )
        entrants.append(
            {
                **common,
                "entrant_id": f"run:{run.run_id}+structured",
                "label": f"{name} or MDL tracer {MDL_TRACER_VERSION}",
                "algorithm": f"SvgFormer {size} or MDL tracer {MDL_TRACER_VERSION}",
                "description": (
                    f"Per image, {model}; or the MDL tracer {MDL_TRACER_VERSION}'s SVG when, drawn over the background "
                    f"it fits, it matches the input {PICK_MARGIN:g} times as closely (squared error)."
                ),
                "settings": {**settings, "paired_with": "structured", "margin": PICK_MARGIN},
            }
        )
    return entrants


def baseline_entrant(key: str) -> dict[str, Any]:
    label, description, settings = ENTRANTS[key]
    return {"entrant_id": f"baseline:{key}", "label": label, "description": description, "settings": settings}


def record_for(
    protocol_name: str, entrant: dict[str, Any], targets: list[Target], hardware: dict[str, Any]
) -> EvaluationRecord:
    label, description, settings = entrant["label"], entrant["description"], entrant["settings"]
    run: RunInfo | None = entrant.get("run")
    key = entrant["entrant_id"].removeprefix("baseline:")
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        ran = pool.submit(run_entrant, settings, targets, entrant.get("weights")).result()
    images = [
        {**score(target, svg), "image_us": us, "svg": svg}
        for target, svg, us in zip(targets, ran["svgs"], ran["image_us"], strict=True)
    ]
    summary, by_set = summarize_sets(images)
    fidelity = [image["fidelity"] for image in images]
    sets = PROTOCOLS[protocol_name]
    peak, baseline = ran["peak_rss_bytes"], ran["baseline_rss_bytes"]
    training = {"measured": True, "none": True}
    if run:
        training = dict((run.summary or {}).get("cost") or {"measured": False})
    return EvaluationRecord(
        game=GAME,
        protocol=protocol_name,
        entrant_id=entrant["entrant_id"],
        entrant_kind="champion" if run else "baseline",
        label=label,
        interface=INTERFACE,
        run_id=run.run_id if run else None,
        champion_ref=entrant.get("champion_ref"),
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
                "parameters": entrant.get("parameters", 0),
                "artifact_bytes": round(summary["bytes"]),
                **({"model_bytes": len(entrant["weights"]), "device": ran["device"]} if run else {}),
            },
            "training": training,
            "model": {
                "description": description,
                "observer_level": 1,
                "note": (run.config or {}).get("note") if run else None,
                "shape": {"algorithm": entrant.get("algorithm", label)},
            },
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
                **({"version": MDL_TRACER_VERSION} if key == "structured" or settings.get("paired_with") else {}),
                **({"picked_tracer": ran["chosen"].count("tracer")} if ran.get("chosen") else {}),
                "columns": summary,
                **({"by_set": by_set} if len(sets) > 1 else {}),
                "by_severity": by_severity(images),
                "images": images,
                "memory": {
                    "peak_rss_bytes": peak,
                    "baseline_rss_bytes": baseline,
                    "peak_above_baseline_bytes": peak - baseline if peak and baseline else None,
                    **({"peak_gpu_bytes": ran["peak_gpu_bytes"]} if ran.get("peak_gpu_bytes") else {}),
                },
            },
        },
        hardware={**hardware, "engine": "torch" if run else "rust", **({"gpu": ran["gpu"]} if ran.get("gpu") else {})},
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
    entrants = [*(baseline_entrant(key) for key in ENTRANTS), *svgformer_entrants(sink)]
    records = [record_for(protocol_name, entrant, targets, hardware) for entrant in entrants]
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
