"""`copycat.report`: the structured pipeline measured for one change (docs/design/0022), recorded under
`<data dir>/experiments/`, and compared image by image with an earlier report.

The leaderboard (`copycat.clean.v1`) scores only `synthetic`'s held-out images, which is the right public number and
too little to develop against. So this measures, and keeps per image:

- `synthetic/train` and `icons/train`, what a change is tuned on;
- `icons/train-control`, the same icons at the `clean` severity: what shapes and layers cost without corruption;
- `synthetic/validate`, 400 more seeds: enough to see a change that moves few images, and free to debug on;
- with `heldout: true`, both held-out sets. Those are for scoring once per change, at the end, never for tuning.

The `train` and `validate` images are built on first use (`corrupt.v1` is seeded, so they are always the same) and kept with the reports.
Each image's SVG goes beside the report (`<out>.svgs.json`), so a change can be looked at later as it was (docs/design/0023).
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import subprocess
import time
from pathlib import Path
from typing import Any

from arena import copycat
from arena.copycat import compare
from arena.copycat_sets import group_targets
from jobcore import Sink, experiments_dir
from jobcore.specs import EvalSpec

from evaluator.registry import protocol

PROTOCOL = "copycat.report"
HELD_OUT = ("synthetic/heldout", "icons/heldout")
TRAIN = ("synthetic/train", "icons/train", "icons/train-control", "synthetic/validate")
COLUMNS = ("fidelity", "alpha_iou", "layer_recovery", "shape_ratio", "palette_ratio", "smoothness_ratio")


def targets_of(group: str) -> list[copycat.Target]:
    return group_targets(group, experiments_dir() / "copycat-sets")


def score_image(target: copycat.Target) -> tuple[dict[str, Any], str]:
    from games.copycat import trace

    svg = trace(target.rgba8(), target.width, target.height).canonical()
    return copycat.score(target, svg), svg


def measure(group: str, pool: concurrent.futures.Executor) -> tuple[dict[str, Any], dict[str, str]]:
    targets = targets_of(group)
    images, svgs = zip(*pool.map(score_image, targets, chunksize=1), strict=True)
    summary = copycat.summarize(images)
    row = {
        "mean": summary["fidelity"],
        "worst": min(image["fidelity"] for image in images),
        "columns": {column: summary[column] for column in COLUMNS},
        "by_severity": {s: row["fidelity"] for s, row in copycat.by_severity(images).items()},
        "per_image": {str(t.seed): round(i["fidelity"], 3) for t, i in zip(targets, images, strict=True)},
    }
    return row, {str(t.seed): svg for t, svg in zip(targets, svgs, strict=True)}


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
        both = {group: measure(group, pool) for group in groups}
    measured = {group: row for group, (row, _) in both.items()}
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "commit": commit(),
        "note": params.note,
        "written_at": time.time(),
        "groups": measured,
    }
    if params.decision:
        result["decision"] = params.decision
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
        svgs = {group: images for group, (_, images) in both.items()}
        (path.parent / f"{params.out}.svgs.json").write_text(json.dumps(svgs), encoding="utf-8")
    return result
