"""Golden runs: a small, deterministic run of every training job, reduced to a digest (docs/design/0018, stage 2).

Recorded from the original jobs/*_run.py scripts before anything was ported, so each trainer adapter has to reproduce
its script exactly: the same recorded config, the same per-generation curve and the same champion bytes. Every
learner here is deterministic per seed, so this is an equality test, not a tolerance. `test_golden.py` re-runs every
case against the fixture; when a case moves to the trainer, only its `run` changes, never the fixture.

Exact *per operating system*, not across them: `math.tanh`/`exp` (Python) and Rust's `f64` functions call the
platform's libm, and MSVC's and glibc's can differ in the last bit -- which a few generations of selection turn into
different champions. So there is one fixture per platform (`golden/champions.<sys.platform>.json`). TinyLM is the one
tolerance: its matrix products go through numpy's BLAS, whose kernels (so whose rounding) depend on the CPU, so its
weights are compared as per-array sums to 1e-9 (relative, or absolute near zero).

Re-record (only for a deliberate behaviour change, which then needs saying in the commit):
    uv run python jobs/tests/golden_cases.py --record [case ...]
A platform without a fixture (Linux, recorded by CI): when `REDQUEEN_GOLDEN_OUT` is set, `test_golden.py` writes the
digests it got there, and CI uploads that directory as the `golden-<platform>` artifact when the test fails.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent / "golden" / f"champions.{sys.platform}.json"
TOLERANT = {"tinylm"}  # cases compared with `close()`, not ==
JOBS = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    """Either a TrainSpec run by the trainer (`spec`: a ported job), or a script not yet ported: `module.main(**kwargs)`
    with `patches` (module attribute -> value) applied first. A ported case's spec is what the script's kwargs and
    patches meant, so its digest is unchanged."""

    module: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    patches: dict[str, Any] = field(default_factory=dict)  # "module.ATTR" -> value
    spec: dict[str, Any] | None = None


def train(game: str, algorithm: str, generations: int, **fields: Any) -> Case:
    return Case(spec={"game": game, "algorithm": algorithm, "budget": {"generations": generations}, **fields})


CASES: dict[str, Case] = {
    "gp": train("symbolic-regression", "gp", 8),
    "snake-neuro-lexicase": train("snake", "neuroevolution", 3, held_out_every=2, params={"population_size": 8}),
    "snake-neuro-tournament-egocentric": train(
        "snake",
        "neuroevolution",
        3,
        interface="snake/egocentric.v1+relative3.v1",
        held_out_every=2,
        seed=5,
        params={"population_size": 8, "seeds": "resample:3", "selection": "tournament"},
    ),
    "snake-neat": train("snake", "neat", 3, held_out_every=2, params={"population_size": 12}),
    "checkers-neuro": train(
        "checkers",
        "neuroevolution",
        2,
        held_out_every=1,
        params={"population_size": 6, "hidden": 4, "opponents": ["random", "material-1"], "monitor_games": 2},
    ),
    "checkers-neuro-hall-depth2": train(
        "checkers",
        "neuroevolution",
        2,
        held_out_every=1,
        params={
            "population_size": 6,
            "hidden": 4,
            "opponents": ["random", "material-1"],
            "monitor_games": 2,
            "depth": 2,
            "hall": 2,
            "seed_material": 0.5,
            "margin": True,
            "resample": True,
        },
    ),
    "checkers-neat": train(
        "checkers",
        "neat",
        2,
        held_out_every=1,
        params={
            "population_size": 8,
            "opponents": ["random", "material-2"],
            "monitor_games": 2,
            "depth": 2,
            "hall": 2,
            "seed_material": 0.5,
        },
    ),
    "checkers-distill": train(
        "checkers",
        "distill",
        2,
        held_out_every=1,
        params={
            "population_size": 6,
            "hidden": 4,
            "depth": 1,
            "label_depth": 2,
            "pool_games": 4,
            "sample": 20,
            "opponents": ["random"],
            "monitor_games": 2,
        },
    ),
    "bandit-evolve": train(
        "bandit",
        "bandit_evolve",
        3,
        params={"scenarios": ["classic", "two-lamps"], "population_size": 6, "games_per_scenario": 10},
    ),
    **{
        f"rl-{algorithm}": Case(
            "rl_run",
            {"algorithm": algorithm, "iterations": 2, "steps_per_iteration": 3_000, "held_out_every": 1, "rng_seed": 2},
            {"rl_run.MONITOR_SEEDS": (20_000, 20_001)},
        )
        for algorithm in ("random", "q_learning", "sarsa", "dqn", "reinforce", "a2c", "ppo")
    },
    "rl-ppo-reach1d": Case(
        "rl_run", {"algorithm": "ppo", "env_id": "reach1d", "iterations": 2, "steps_per_iteration": 3_000}
    ),
    "selfplay-td": Case(
        "checkers_selfplay_run",
        {
            "iterations": 3,
            "games_per_iteration": 10,
            "depth": 1,
            "held_out_every": 2,
            "rng_seed": 3,
            "params": {"hidden": 8, "hidden_layers": 1},
        },
        {"checkers_selfplay_run.MONITOR_GAMES": 2},
    ),
    "selfplay-td-leaf": Case(
        "checkers_selfplay_run",
        {
            "iterations": 2,
            "games_per_iteration": 4,
            "depth": 1,
            "held_out_every": 2,
            "rng_seed": 3,
            "params": {"hidden": 8, "hidden_layers": 1, "search_depth": 2},
        },
        {"checkers_selfplay_run.MONITOR_GAMES": 2},
    ),
    "selfplay-alphazero": Case(
        "checkers_selfplay_run",
        {
            "iterations": 2,
            "games_per_iteration": 2,
            "depth": 1,
            "held_out_every": 2,
            "rng_seed": 3,
            "params": {"hidden": 8, "hidden_layers": 1, "simulations": 4},
            "algorithm": "alphazero",
        },
        {"checkers_selfplay_run.MONITOR_GAMES": 2, "checkers_selfplay_run.MCTS_GAMES": 2},
    ),
    "pbt": Case(
        "checkers_pbt_run",
        {"members": 4, "games": 40, "interval": 20, "depth": 1, "openings_per_pair": 1, "rng_seed": 3},
        {"checkers_pbt_run.BASE": {"hidden": 4, "pool_every": 20, "pool_size": 2}},
    ),
    # A pipeline (docs/design/0018: scheduler): a TD run, then a TD-Leaf fine-tune continuing its network.
    "recipe": Case(
        "checkers_recipe_run",
        {
            "hidden": 8,
            "layers": 1,
            "td_games": 40,
            "leaf_games": 4,
            "leaf_depth": 2,
            "depth": 1,
            "games_per_iteration": 20,
        },
        {"checkers_selfplay_run.MONITOR_GAMES": 2},
    ),
    "tinylm": Case("tinylm_run", {"name": "golden"}, {"tinylm_run.STEPS": 3}),
}


@contextlib.contextmanager
def _patched(patches: dict[str, Any]) -> Iterator[None]:
    saved = []
    try:
        for target, value in patches.items():
            module_name, attr = target.rsplit(".", 1)
            module = importlib.import_module(module_name)
            saved.append((module, attr, getattr(module, attr)))
            setattr(module, attr, value)
        yield
    finally:
        for module, attr, value in reversed(saved):
            setattr(module, attr, value)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# A Python repr's memory address (the GP prototype stores its champion as a repr, function objects included).
ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")


def _anonymize(value: Any, names: dict[str, str]) -> Any:
    """Run ids are random: every one a case created (and its 8-character short form, as notes use) is replaced by
    `<run0>`, `<run1>`, ... in creation order -- so a fine-tune's `init_run` still has to name the right parent."""
    if isinstance(value, str):
        for run_id, name in names.items():
            value = value.replace(run_id, name).replace(run_id[:8], name)
        return value
    if isinstance(value, list):
        return [_anonymize(v, names) for v in value]
    if isinstance(value, dict):
        return {k: _anonymize(v, names) for k, v in value.items()}
    return value


def digest_run(run_id: str, names: dict[str, str]) -> dict[str, Any]:
    """What a run must reproduce: its config, its curve (everything but wall-clock timestamps) and its final champion's
    bytes. The summary is left out: it holds measured costs (seconds, memory), which are clocks, not results."""
    from telemetry import open_stores

    stores = open_stores()
    run = stores.registry.get_run(run_id)
    history = stores.metrics.history(run_id)
    curve = [_anonymize(g.model_dump(exclude={"timestamp", "run_id"}), names) for g in history]
    champion = stores.artifacts.get_program(history[-1].champion_ref) if history[-1].champion_ref else b""
    return {
        "status": run.status,
        "config": _anonymize(run.config, names),
        "curve_sha256": _sha(json.dumps(curve, sort_keys=True).encode()),
        "generations": len(history),
        "final": curve[-1],
        "champion_sha256": _sha(ADDRESS.sub(" at 0x?", champion.decode("utf-8", "replace")).encode()),
    }


def digest_tinylm(name: str) -> dict[str, Any]:
    """TinyLM isn't a telemetry run: its checkpoint's arrays (not the .npz bytes -- zip entries carry timestamps), as
    each array's shape, sum and sum of squares -- BLAS rounding differs by CPU, so these are compared to 1e-9."""
    import numpy as np
    from telemetry import data_dir

    path = data_dir() / "tinylm" / name
    with np.load(path.with_suffix(".npz")) as arrays:
        weights = {
            k: {"shape": list(arrays[k].shape), "sum": float(arrays[k].sum()), "sumsq": float((arrays[k] ** 2).sum())}
            for k in sorted(arrays.files)
        }
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    meta.get("training", {}).pop("seconds", None)
    return {"weights": weights, "meta": meta}


def close(actual: Any, expected: Any, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Equal, except that floats need only agree to `rel` -- or to `abs_tol` near zero, where a relative tolerance means
    nothing: an attention key bias's gradient is exactly zero in theory (softmax ignores a constant shift), so its sum
    is ~1e-13 of pure rounding noise that differs from CPU to CPU (TOLERANT cases)."""
    if isinstance(expected, float) and isinstance(actual, float):
        return abs(actual - expected) <= max(rel * abs(expected), abs_tol)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(close(actual[k], expected[k], rel, abs_tol) for k in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            close(a, e, rel, abs_tol) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected


def run_case(name: str, runner: Callable[[Case], Any] | None = None) -> dict[str, Any]:
    """Runs one case in a fresh data directory and returns its digest. `runner` replaces calling the old script's
    `main` -- how a ported adapter is checked against the same fixture."""
    case = CASES[name]
    if case.spec is not None:
        from trainer import train as run_spec

        runner = runner or (lambda c: run_spec(c.spec))
    else:
        importlib.import_module(case.module)  # bound before patching, so a patch always lands on the live binding
    with tempfile.TemporaryDirectory() as scratch, _env("REDQUEEN_DATA_DIR", scratch), _patched(case.patches):
        result = runner(case) if runner else importlib.import_module(case.module).main(**case.kwargs)
        if case.module == "tinylm_run":
            return digest_tinylm(case.kwargs["name"])
        run_ids = result if isinstance(result, tuple) else (result,)
        names = {run_id: f"<run{i}>" for i, run_id in enumerate(run_ids)}
        return {"runs": [digest_run(run_id, names) for run_id in run_ids]}


@contextlib.contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def record(names: list[str]) -> None:
    fixture = load_fixture() if FIXTURE.exists() else {}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"recording {name} ...", flush=True)
        fixture[name] = run_case(name)
        # Saved after every case, so one failing case doesn't discard the ones before it.
        text = json.dumps(dict(sorted(fixture.items())), indent=1, sort_keys=True) + "\n"
        FIXTURE.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(JOBS))
    args = sys.argv[1:]
    if not args or args[0] != "--record":
        sys.exit(__doc__)
    record(args[1:] or list(CASES))
