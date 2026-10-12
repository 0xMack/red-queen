"""Golden evaluations and publishes: every leaderboard protocol, report and publish job, on a small fixed field,
reduced to a digest (docs/design/0018 stage 4) -- the evaluator/publisher counterpart of golden_cases.py.

Recorded from the original jobs/evaluate*.py, checkers_sprt.py, checkers_round_robin.py and publish_models.py before
they were ported to the evaluator and publisher workloads, which now reproduce them. A case builds the field (`FIELD`: small trainer runs, themselves pinned by golden_cases.py) in a fresh
data directory, runs its job, and digests what it wrote: evaluation records without their clocks (created_at, hardware,
latencies, measured seconds), a report as-is, a model store's catalog and manifests without content hashes (parity is
measured by ONNX Runtime, whose kernels -- so rounding -- vary by CPU). Exact per OS, like golden_cases.py.

Re-record (only for a deliberate behaviour change):
    uv run python jobs/tests/golden_eval.py --record [case ...]
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from golden_cases import _anonymize, _env, _patched, close

FIXTURE = Path(__file__).parent / "golden" / f"evaluations.{sys.platform}.json"
JOBS = Path(__file__).resolve().parents[1]

# The field every case evaluates: one of each kind of entrant a leaderboard or the model store takes.
FIELD: list[dict[str, Any]] = [
    {"game": "snake", "algorithm": "neuroevolution", "budget": {"generations": 3}, "params": {"population_size": 8}},
    *(
        {
            "game": "checkers",
            "algorithm": "td_lambda",
            "budget": {"iterations": 10},  # evaluate_versus.MIN_GENERATIONS: shorter runs are smoke tests, not entrants
            "seed": seed,
            "params": {"games_per_iteration": 5, "depth": 1, "monitor_games": 2, "agent": {"hidden": 8}},
        }
        for seed in (0, 1)
    ),
    {"game": "bandit", "algorithm": "bandit_evolve", "budget": {"generations": 3}, "params": {"population_size": 6}},
    {"game": "text", "algorithm": "tinylm", "budget": {"steps": 3}, "params": {"checkpoint": "golden"}},
]

SNAKE_20 = tuple(range(10_000, 10_020))  # 20 held-out games, not 200: the protocol's rules, a test's budget


@dataclass(frozen=True)
class EvalCase:
    """`run()` (the job, as a script function or the ported workload) then `digest()` of what it wrote."""

    run: Callable[[], Any]
    digest: Callable[[Any], Any]
    patches: dict[str, Any] = field(default_factory=dict)


def _records(game: str) -> Callable[[Any], Any]:
    def digest(_result: Any) -> Any:
        from telemetry import open_stores

        # sorted by entrant once anonymized (run_case): raw run ids are random, so sorting by them is too
        return [_strip_clocks(r.model_dump(mode="json")) for r in open_stores().evaluations.list(game)]

    return digest


CLOCKS = {"created_at", "hardware", "wall_s", "active_s", "cpu_s", "paused_s", "peak_rss_bytes", "seconds"}


def _strip_clocks(value: Any) -> Any:
    """Clocks dropped; a long list of numbers (every game's score) kept as its length, sum and hash -- still exact."""
    if isinstance(value, dict):
        return {k: _strip_clocks(v) for k, v in value.items() if k not in CLOCKS and not k.endswith("_us")}
    if isinstance(value, list):
        if len(value) > 32 and all(isinstance(v, int | float) for v in value):
            text = json.dumps(value).encode()
            return {"n": len(value), "sum": sum(value), "sha256": hashlib.sha256(text).hexdigest()}
        return [_strip_clocks(v) for v in value]
    return value


def _report(result: Any) -> Any:
    return result


def _store(game: str) -> Callable[[Any], Any]:
    """The model store's catalog for `game` and every manifest it lists, without content hashes (see the module
    docs); parity numbers are kept and compared with a tolerance (`close`)."""

    def digest(_result: Any) -> Any:
        from modelpack import LocalModelStore
        from telemetry import models_dir

        store = LocalModelStore(models_dir())
        entries = []
        for entry in store.catalog(game).entries:
            manifest = json.loads(store.manifest_path(entry.package_id).read_text(encoding="utf-8"))
            entries.append(
                {
                    "entry": _drop(entry.model_dump(mode="json"), {"package_id"}),
                    "manifest": _drop(manifest, {"package_id", "sha256", "created_at", "blobs", "files", "uri"}),
                }
            )
        return sorted(entries, key=lambda e: e["entry"]["entrant_id"])

    return digest


def _drop(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _drop(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_drop(v, keys) for v in value]
    return value


def _evaluate(protocol: str, **params: Any) -> Callable[[], Any]:
    def run() -> Any:
        from evaluator import evaluate

        return evaluate({"protocol": protocol, "params": params})

    return run


def _publish(target: str) -> Callable[[], Any]:
    def run() -> Any:
        from publisher import publish

        return publish({"target": target})

    return run


EVAL_CASES: dict[str, EvalCase] = {
    "snake.score.v2": EvalCase(
        _evaluate("snake.score.v2"),
        _records("snake"),
        {"arena.snake.HELD_OUT_SEEDS": SNAKE_20, "evaluator.protocols.snake_score.HELD_OUT_SEEDS": SNAKE_20},
    ),
    "checkers.versus.v2": EvalCase(_evaluate("checkers.versus.v2"), _records("checkers")),
    "bandit.skill.v1": EvalCase(_evaluate("bandit.skill.v1"), _records("bandit")),
    # the result as the script wrote it with --out: what a report returns
    "checkers.sprt": EvalCase(_evaluate("checkers.sprt", a="material-2", b="random"), _report),
    "checkers.static-round-robin": EvalCase(_evaluate("checkers.static-round-robin"), _report),
    "publish.snake": EvalCase(
        _publish("snake"),
        _store("snake"),
        {"arena.snake.HELD_OUT_SEEDS": SNAKE_20, "publisher.models.HELD_OUT_SEEDS": SNAKE_20},
    ),
    "publish.tinylm": EvalCase(_publish("tinylm"), _store("tinylm")),
}
TOLERANT = {"publish.snake", "publish.tinylm"}  # ONNX Runtime parity numbers


def build_field() -> list[str]:
    from trainer import train

    return [train(spec) for spec in FIELD]


def run_case(name: str, case: EvalCase | None = None) -> Any:
    case = case or EVAL_CASES[name]
    with tempfile.TemporaryDirectory() as scratch, _env("REDQUEEN_DATA_DIR", scratch):
        run_ids = build_field()
        with _patched(case.patches), contextlib.redirect_stdout(sys.stderr):
            result = case.run()
        names = {run_id: f"<run{i}>" for i, run_id in enumerate(run_ids)}
        digest = _anonymize(case.digest(result), names)
        if isinstance(digest, list) and all(isinstance(d, dict) and "entrant_id" in d for d in digest):
            digest.sort(key=lambda d: d["entrant_id"])
        return digest


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def record(names: list[str]) -> None:
    fixture = load_fixture() if FIXTURE.exists() else {}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"recording {name} ...", flush=True)
        fixture[name] = run_case(name)
        text = json.dumps(dict(sorted(fixture.items())), indent=1, sort_keys=True) + "\n"
        FIXTURE.write_text(text, encoding="utf-8")


def matches(name: str, actual: Any, expected: Any) -> bool:
    return close(actual, expected, rel=1e-3, abs_tol=1e-6) if name in TOLERANT else actual == expected


if __name__ == "__main__":
    sys.path.insert(0, str(JOBS))
    args = sys.argv[1:]
    if not args or args[0] != "--record":
        sys.exit(__doc__)
    record(args[1:] or list(EVAL_CASES))
