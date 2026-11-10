"""Experiment E0 (docs/design/0024): what the tokens cost before any learning.

Every clean SVG of a set is tokenized and decoded, and the decoded SVG is scored against the original exactly as
`copycat.clean.v2` scores an entrant (`arena.copycat.score`): so its fidelity is the ceiling no model writing these tokens
can pass. Also each sequence's length, and how many numbers fell outside the grid.

    uv run python -m svgformer.ceilings [--root DATA/experiments/copycat-sets] [--artwork DATA/experiments/svgformer-artwork]

With `--artwork`, the training artwork (`svgformer.build`) too: icons at 128 px, illustrations at 256.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from arena.copycat import Target, score
from arena.copycat_sets import SPLITS, synthetic

from svgformer import tokens as T

SIZE = 128


def measure(name: str, svgs: list[str], size: int = SIZE) -> dict[str, float]:
    fidelity, lengths, clamped = [], [], 0
    for svg in svgs:
        encoded = T.encode(svg, size)
        decoded = T.decode(encoded.tokens, size)
        target = Target(id=name, seed=0, severity="clean", levels={}, clean=svg, png=b"", width=size, height=size)
        fidelity.append(score(target, decoded)["fidelity"])
        lengths.append(len(encoded.tokens))
        clamped += encoded.clamped
    lengths.sort()
    return {
        "images": len(svgs),
        "fidelity_mean": statistics.fmean(fidelity),
        "fidelity_min": min(fidelity),
        "length_median": statistics.median(lengths),
        "length_p95": lengths[int(0.95 * (len(lengths) - 1))],
        "length_max": lengths[-1],
        "under_1024": sum(n <= 1024 for n in lengths) / len(lengths),
        "clamped_numbers": clamped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/experiments/copycat-sets"))
    parser.add_argument("--artwork", type=Path, help="also the training artwork built under this root")
    parser.add_argument(
        "--sample", type=int, default=3000, help="at most this many of each artwork file, evenly spaced"
    )
    args = parser.parse_args()
    sets = {"synthetic": ([synthetic(seed)[0] for split in ("train", "validate") for seed in SPLITS[split]], SIZE)}
    for name in ("icons", "broad"):
        files = sorted((args.root / name / "clean").glob("*.svg"))
        if files:
            sets[name] = ([f.read_text(encoding="utf-8") for f in files], SIZE)
    if args.artwork:
        for stem, size in (("icons", 128), ("illustrations", 256), ("illustrations-heldout", 256)):
            lines = (args.artwork / "build" / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
            step = max(1, len(lines) // args.sample)
            sets[f"artwork:{stem}"] = ([json.loads(line)["svg"] for line in lines[::step]], size)
    columns = ("images", "fidelity_mean", "fidelity_min", "length_median", "length_p95", "length_max", "under_1024")
    print("| set | " + " | ".join(columns) + " | clamped |")
    print("|---" * (len(columns) + 2) + "|")
    for name, (svgs, size) in sets.items():
        row = measure(name, svgs, size)
        cells = [f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        print(f"| {name} | " + " | ".join(cells) + f" | {row['clamped_numbers']} |")


if __name__ == "__main__":
    main()
