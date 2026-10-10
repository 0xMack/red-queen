"""Run one training spec: `uv run python -m trainer SPEC [--set key=value ...]`.

SPEC is a .yaml/.json file (jobs/trainer/specs/ has one per algorithm, at the original scripts' defaults). `--set`
overrides a field before validation: `--set seed=3`, `--set budget.generations=20`, `--set params.hidden=32`,
`--set tags.experiment=probe` -- the value parsed as YAML, so numbers and lists work (`--set params.opponents=[random]`).
Prints the run id; the run itself is in the data directory (telemetry.data_dir()) like any other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from jobcore.specs import TrainSpec

from trainer import train


def _set(data: dict[str, Any], assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise SystemExit(f"--set takes key=value, got {assignment!r}")
    *parents, leaf = key.split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(prog="python -m trainer", description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path, help="a TrainSpec as .yaml/.yml/.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field")
    args = parser.parse_args(argv)
    data = yaml.safe_load(args.spec.read_text(encoding="utf-8"))  # YAML is a superset of JSON
    for assignment in args.set:
        _set(data, assignment)
    try:
        spec = TrainSpec.model_validate(data)
        spec.resolve()
    except ValueError as e:  # pydantic's ValidationError is one: a bad spec is the user's to fix, not a traceback
        raise SystemExit(f"invalid spec {args.spec}: {e}") from None
    run_id = train(spec)
    print(run_id)
    return run_id


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
