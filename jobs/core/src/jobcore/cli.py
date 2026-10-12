"""The command line every workload shares: `uv run python -m <workload> SPEC [--set key=value ...]`.

SPEC is a .yaml/.json file. `--set` overrides a field before validation, with dotted keys (`seed=3`,
`budget.generations=20`, `params.agent.hidden=32`); the value is parsed as YAML, so numbers and lists work. A spec
that doesn't validate is reported as one line, not a traceback -- it's the user's to fix.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from jobcore.specs import Spec, parse_spec


def _set(data: dict[str, Any], assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise SystemExit(f"--set takes key=value, got {assignment!r}")
    *parents, leaf = key.split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)


def run_spec_command(prog: str, kind: str, run: Callable[[Any], Any], argv: list[str] | None = None) -> Any:
    """Parse `argv`, load and override the spec, check it is a `kind` spec that resolves, run it, print the result."""
    parser = argparse.ArgumentParser(
        prog=f"python -m {prog}", description=f"Run one {kind} spec ({__doc__.split(':')[0]})."
    )
    parser.add_argument("spec", type=Path, help="a spec as .yaml/.yml/.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field")
    args = parser.parse_args(argv)
    data = yaml.safe_load(args.spec.read_text(encoding="utf-8"))  # YAML is a superset of JSON
    data.setdefault("kind", kind)
    for assignment in args.set:
        _set(data, assignment)
    try:
        spec: Spec = parse_spec(data)
        if spec.kind != kind:
            raise ValueError(f"this is the {kind} workload; the spec is a {spec.kind} spec")
        spec.resolve()
    except ValueError as e:  # pydantic's ValidationError is one
        raise SystemExit(f"invalid spec {args.spec}: {e}") from None
    result = run(spec)
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))
    return result
