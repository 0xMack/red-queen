"""Is Checkers player A stronger than player B? A head-to-head SPRT over the ballot (docs/design/0013).

A and B play game pairs (one ballot opening, each from both seats; jobs/evaluate_versus.py's `play_pairing`) until
Wald's sequential test is conclusive (libs/arena/src/arena/versus_stats.py's `Sprt`): H1, A is at least `elo1` stronger, or H0, A is
no stronger than `elo0`. The field doesn't enter into it, so it keeps working however strong the players get. The
openings are played in a fixed shuffled order, so an early stop isn't a verdict on one family of openings. If the
ballot runs out first (174 pairs) the answer is *inconclusive*: the difference is smaller than the test can resolve,
and the Elo estimate and its interval say how small.

A player is a fixed strategy (`material-4`, `random`, ...) or a run's final champion: its run id, or any unique
prefix of it, optionally `@N` to search N plies instead of the depth the run was trained for.

  uv run python jobs/checkers_sprt.py A B [--elo0 0] [--elo1 50] [--alpha 0.05] [--beta 0.05]
  e.g. uv run python jobs/checkers_sprt.py 672890ba@4 material-4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Sequence
from typing import Any

from arena.checkers import SPRT_ORDER_SEED, SPRT_SEED_BASE, Factory, pair_points, play_pairing
from arena.versus_stats import Sprt
from evolve import NeatGenome, network_from_json
from games.checkers_openings import Opening, ballot
from games.checkers_strategies import STRATEGIES, evaluator, graph_evaluator, strategy
from run_context import experiments_dir
from telemetry import FileArtifactStore, FileMetricsStore, RunInfo, SqliteRunRegistry, open_stores


def sprt_order() -> list[Opening]:
    """The ballot in the fixed shuffled order a sequential test plays it."""
    openings = list(ballot())
    random.Random(SPRT_ORDER_SEED).shuffle(openings)
    return openings


def run_factory(run: RunInfo, metrics: FileMetricsStore, artifacts: FileArtifactStore, depth: int | None = None):
    """A run's final champion as a player, searching `depth` plies (default: the depth the run recorded)."""
    raw = artifacts.get_program(metrics.history(run.run_id)[-1].champion_ref).decode("utf-8")
    champion = network_from_json(raw)
    depth = depth or int(run.config.get("search_depth", 1))
    if isinstance(champion, NeatGenome):
        return graph_evaluator(champion.graph_encoding(), depth)
    return evaluator(champion.weights, champion.layer_sizes, depth)


def resolve(spec: str, registry: SqliteRunRegistry, metrics: FileMetricsStore, artifacts: FileArtifactStore) -> Factory:
    """`material-4` / `baseline:material-4` / a run id or unique prefix, optionally `@depth`."""
    name = spec.removeprefix("baseline:")
    if name in STRATEGIES or name.startswith("material-"):
        return strategy(name)
    run_id, _, depth = spec.removeprefix("run:").partition("@")
    matches = [r for r in registry.list_runs() if r.run_id.startswith(run_id)]
    if len(matches) != 1:
        sys.exit(f"{spec!r} matches {len(matches)} runs")
    return run_factory(matches[0], metrics, artifacts, int(depth) if depth else None)


def sequential_match(
    players: Sequence[tuple[Factory, Factory]],
    test: Sprt,
    openings: Sequence[Opening] | None = None,
    seed_base: int = SPRT_SEED_BASE,
    on_pair: Callable[[int, list[float]], None] | None = None,
) -> dict[str, Any]:
    """Play `(a, b)` game pairs opening by opening until `test` decides or the openings run out. `players` may hold
    several (a, b) couples -- an experiment's arm against its baseline, one couple per training seed -- in which case
    every opening is played by each couple before the next, and every pair feeds the one test. Returns the test's
    summary plus the pairs each couple scored (from a's side)."""
    openings = list(openings if openings is not None else sprt_order())
    per_couple: list[list[float]] = [[] for _ in players]
    for k, opening in enumerate(openings):
        scored = []
        for c, (a, b) in enumerate(players):
            games = play_pairing(a, b, [opening], seed_base + 2 * (k * len(players) + c))
            (points,) = pair_points(games)
            per_couple[c].append(points)
            test.add(points)
            scored.append(points)
        if on_pair:
            on_pair(k, scored)
        if test.verdict:
            break
    return {
        **test.summary(),
        "openings_played": len(per_couple[0]),
        "openings_available": len(openings),
        "per_couple": per_couple,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Head-to-head SPRT between two Checkers players (docs/design/0013).")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=50.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--out", default=None, help="write the result as JSON under data/experiments/")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    stores = open_stores()
    registry, metrics, artifacts = stores.registry, stores.metrics, stores.artifacts
    a = resolve(args.a, registry, metrics, artifacts)
    b = resolve(args.b, registry, metrics, artifacts)
    test = Sprt(elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta)

    def progress(k: int, scored: list[float]) -> None:
        if (k + 1) % 10 == 0 or test.verdict:
            print(f"pair {k + 1:>3}  LLR {test.llr:+.2f} in [{test.lower:.2f}, {test.upper:.2f}]", flush=True)

    result = sequential_match([(a, b)], test, on_pair=progress)
    result = {"a": args.a, "b": args.b, **result}
    del result["per_couple"]
    print(
        f"{args.a} vs {args.b}: {result['verdict']} after {result['pairs']} pairs -- "
        f"Elo {result['elo']:+.0f} [{result['lo']:+.0f}, {result['hi']:+.0f}], pentanomial {result['pentanomial']}"
    )
    if args.out:
        path = experiments_dir() / f"{args.out}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
