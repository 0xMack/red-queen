"""`copycat.report`: the structured pipeline measured for one change (docs/design/0022), recorded under
`<data dir>/experiments/`, and compared image by image with an earlier report.

The leaderboard (`copycat.clean.v1`) scores only `synthetic`'s held-out images, which is the right public number and
too little to develop against. So this measures, and keeps per image:

- `synthetic/train` and `icons/train`, what a change is tuned on;
- `icons/train-control`, the same icons at the `clean` severity: what shapes and layers cost without corruption;
- with `heldout: true`, both held-out sets. Those are for scoring once per change, at the end, never for tuning.

The `train` images are built on first use (`corrupt.v1` is seeded, so they are always the same) and kept with the reports.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import multiprocessing
import subprocess
from pathlib import Path
from typing import Any

from arena import copycat
from arena.copycat_sets import corrupt, write_split
from jobcore import Sink, experiments_dir
from jobcore.specs import EvalSpec

from evaluator.registry import protocol

PROTOCOL = "copycat.report"
HELD_OUT = ("synthetic/heldout", "icons/heldout")
TRAIN = ("synthetic/train", "icons/train", "icons/train-control")
COLUMNS = ("fidelity", "alpha_iou", "layer_recovery", "shape_ratio", "palette_ratio", "smoothness_ratio")


def targets_of(group: str) -> list[copycat.Target]:
    set_name, split = group.split("/")
    if split == "heldout":
        return copycat.load_set(set_name=set_name)
    root = experiments_dir() / "copycat-sets" / set_name / "train"
    if not (root / "manifest.json").exists():
        write_split("train", root, set_name)
    targets = copycat.load_set("train", root=root)
    if split == "train-control":
        return [
            dataclasses.replace(t, png=corrupt(t.clean, t.seed, "clean")[0], severity="clean", levels={})
            for t in targets
        ]
    return targets


def score_image(target: copycat.Target) -> dict[str, Any]:
    from games.copycat import trace

    return copycat.score(target, trace(target.rgba8(), target.width, target.height).canonical())


def measure(group: str, pool: concurrent.futures.Executor) -> dict[str, Any]:
    targets = targets_of(group)
    images = list(pool.map(score_image, targets, chunksize=1))
    summary = copycat.summarize(images)
    return {
        "mean": summary["fidelity"],
        "worst": min(image["fidelity"] for image in images),
        "columns": {column: summary[column] for column in COLUMNS},
        "by_severity": {s: row["fidelity"] for s, row in copycat.by_severity(images).items()},
        "per_image": {str(t.seed): round(i["fidelity"], 3) for t, i in zip(targets, images, strict=True)},
    }


def compare(before: dict[str, Any], after: dict[str, Any], moved: float = 0.05) -> dict[str, Any]:
    """Each group in both reports: the means, and the images that moved by more than `moved` dB, largest first."""
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


def commit() -> str | None:
    """HEAD, marked if the working tree differs from it: a report of uncommitted code says so."""
    try:
        here = Path(__file__).parent
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


@protocol(PROTOCOL)
def run(spec: EvalSpec, params, sink: Sink) -> dict[str, Any]:
    groups = (*TRAIN, *HELD_OUT) if params.heldout else TRAIN
    with concurrent.futures.ProcessPoolExecutor(
        params.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        measured = {group: measure(group, pool) for group in groups}
    result: dict[str, Any] = {"protocol": PROTOCOL, "commit": commit(), "note": params.note, "groups": measured}
    for group, row in measured.items():
        print(f"{group}: {row['mean']:.3f} dB, worst {row['worst']:.2f}")
    if params.baseline:
        before = json.loads((experiments_dir() / f"{params.baseline}.json").read_text(encoding="utf-8"))
        result["baseline"] = params.baseline
        result["compared"] = compare(before["groups"], measured)
        for group, row in result["compared"].items():
            print(f"{group}: {row['before']:.3f} -> {row['after']:.3f}, {row['up']} up, {row['down']} down")
    if params.out:
        path = experiments_dir() / f"{params.out}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
