"""Tracked comparison of Checkers self-play variants (docs/design/0010 Phase 4, Decision 4).

Like jobs/rl_experiment.py, but a two-player game has no held-out *score* to rank by -- only opponents. Every arm
trains from the same rng seeds for the same number of self-play games (jobs/checkers_selfplay_run.py), each run
tagged `config.experiment` / `config.arm`, and the report plays each run's *final* network, searching `DEPTH`
plies, against a fixed field: material search to 2, 3 and 4 plies, and the strongest evolved evaluator on the
versus leaderboard (a 32 -> 16 -> 1 network searching 3 plies). A run's score is points per game over the field
(win 1, draw 1/2, loss 0; `GAMES_PER_OPPONENT` games each, seats alternating); arms are compared with an exact
paired permutation test against their baseline arm.

Arms (200k self-play games each; 32 -> 16 -> 1 tanh; lambda 0.7 unless noted):
- `sp`            pure self-play
- `sp-pool`       + an opponent pool: half the games against one of the last 10 frozen selves (one every 5k games)
- `sp-lambda0`    one-step TD (lambda 0)
- `sp-lambda1`    Monte-Carlo returns (lambda 1)

`selfplay-v2` scales the pool recipe: `pool-1m` (5x the games), `pool-h64` (a 64-wide hidden layer),
`pool-h64-1m` (both), `pool-2x64-1m` (two 64-wide hidden layers, 1M games), and two controls telling depth from
size: `pool-2x32-1m` (two layers at `pool-h64`'s parameter count) and `pool-h192-1m` (one layer at `pool-2x64-1m`'s).
Report it with `--games 60`: 20 per opponent is too noisy to separate these arms.

`selfplay-v3` is TD-Leaf(λ): `ft-td`, `ft-leaf2`, `ft-leaf3` each continue the same seed's `pool-2x64-1m` network
(from `selfplay-v2`) for 50k games -- plain TD, or self-play searching 2 or 3 plies and learning at the principal
variation's leaves.

  uv run python jobs/checkers_selfplay_experiment.py run    --name NAME --arms sp,sp-pool --seeds 0-4
  uv run python jobs/checkers_selfplay_experiment.py report --name NAME [--games 20]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

import checkers_selfplay_run
from evaluate_versus import play_pairing
from evolve import WeightVector, network_from_json
from games.checkers_strategies import STRATEGIES, evaluator
from rl_experiment import EXPERIMENTS_DIR, paired_permutation_p
from run_context import TelemetryStores
from snake_experiment import _stats, experiment_runs, parse_seeds

DEPTH = 3
GAMES_PER_OPPONENT = 20
SEED_BASE = 40_000  # disjoint from training (self-play draws its own), monitoring (20k) and the leaderboard (30k)
FIELD = ("material-2", "material-3", "material-4")
# The versus leaderboard's best evolved evaluator (0.757 points per game there): the benchmark from the other paradigm.
EVOLVED_RUN = "802e1c7624ae402595d5384aaa1da2e8"


@dataclass(frozen=True)
class Arm:
    params: dict[str, float] = field(default_factory=dict)
    games: int = 200_000
    baseline: str | None = "sp"
    # Continue from the final network of this (experiment, arm)'s run with the same rng seed, instead of a fresh one.
    init: tuple[str, str] | None = None


POOL = {"pool_every": 5000, "pool_size": 10, "pool_fraction": 0.5}
BIG = {**POOL, "hidden": 64, "hidden_layers": 2}
ARMS: dict[str, Arm] = {
    "sp": Arm(baseline=None),
    "sp-pool": Arm(POOL),
    "sp-lambda0": Arm({"lambda": 0.0}),
    "sp-lambda1": Arm({"lambda": 1.0}),
    # selfplay-v2: does more training, or a bigger network, buy strength? (each tested against the arm it extends)
    "pool-1m": Arm(POOL, games=1_000_000, baseline="sp-pool"),
    "pool-h64": Arm({**POOL, "hidden": 64}, baseline="sp-pool"),
    "pool-h64-1m": Arm({**POOL, "hidden": 64}, games=1_000_000, baseline="pool-1m"),
    "pool-2x64-1m": Arm({**POOL, "hidden": 64, "hidden_layers": 2}, games=1_000_000, baseline="pool-h64-1m"),
    # depth, not size? 32 -> 32 -> 32 -> 1 has about as many weights (2.1k) as 32 -> 64 -> 1 (2.2k)
    "pool-2x32-1m": Arm({**POOL, "hidden": 32, "hidden_layers": 2}, games=1_000_000, baseline="pool-h64-1m"),
    # ...or size, not depth? 32 -> 192 -> 1 has about as many weights (6.5k) as 32 -> 64 -> 64 -> 1 (6.3k)
    "pool-h192-1m": Arm({**POOL, "hidden": 192}, games=1_000_000, baseline="pool-2x64-1m"),
    # selfplay-v3: TD-Leaf(λ). Each seed's 2 x 64 champion (selfplay-v2) trains 50k more games, plainly or through
    # its own search -- paired by seed, so the control is the same network given the same games without search.
    "ft-td": Arm(BIG, games=50_000, baseline=None, init=("selfplay-v2", "pool-2x64-1m")),
    "ft-leaf2": Arm({**BIG, "search_depth": 2}, games=50_000, baseline="ft-td", init=("selfplay-v2", "pool-2x64-1m")),
    "ft-leaf3": Arm({**BIG, "search_depth": 3}, games=50_000, baseline="ft-td", init=("selfplay-v2", "pool-2x64-1m")),
}
GAMES_PER_ITERATION = 1000


def run_experiment(name: str, arms: list[str], seeds: list[int]) -> None:
    registry = TelemetryStores.open().registry
    for seed in seeds:
        for arm_name in arms:
            done = [
                r
                for r in experiment_runs(registry, name)
                if r.config.get("arm") == arm_name and r.config.get("rng_seed") == seed and r.status == "completed"
            ]
            if done:
                print(f"skip {arm_name} seed {seed}: already completed as {done[0].run_id}", flush=True)
                continue
            arm = ARMS[arm_name]
            init_run = None
            if arm.init:
                parents = [
                    r
                    for r in experiment_runs(registry, arm.init[0])
                    if r.config.get("arm") == arm.init[1]
                    and r.config.get("rng_seed") == seed
                    and r.status == "completed"
                ]
                if not parents:
                    sys.exit(f"{arm_name} seed {seed} continues {arm.init}, which has no completed run for that seed")
                init_run = parents[0].run_id
            iterations = max(1, arm.games // GAMES_PER_ITERATION)
            print(f"=== {name} · {arm_name} · seed {seed} ({arm.games:,} games)", flush=True)
            checkers_selfplay_run.main(
                iterations=iterations,
                games_per_iteration=GAMES_PER_ITERATION,
                depth=DEPTH,
                params=dict(arm.params),
                rng_seed=seed,
                held_out_every=max(1, iterations // 10),
                tags={"experiment": name, "arm": arm_name},
                init_run=init_run,
            )


def field_factories() -> dict[str, Any]:
    """The fixed opponents: material search, and the best evolved evaluator (if this machine has its run)."""
    factories = {name: STRATEGIES[name] for name in FIELD}
    registry, metrics, artifacts = TelemetryStores.open()
    if EVOLVED_RUN not in {r.run_id for r in registry.list_runs()}:
        return factories
    history = metrics.history(EVOLVED_RUN)
    champion = network_from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    depth = int(registry.get_run(EVOLVED_RUN).config.get("search_depth", 1))
    factories["evolved-3ply"] = evaluator(champion.weights, champion.layer_sizes, depth)
    return factories


def score_run(snapshot: str, field_: dict[str, Any], games: int = GAMES_PER_OPPONENT) -> dict[str, float]:
    """Points per game against each field member, and overall."""
    network = WeightVector.from_json(snapshot)
    me = evaluator(network.weights, network.layer_sizes, DEPTH)
    by_opponent = {}
    for i, (name, opponent) in enumerate(field_.items()):
        played = play_pairing(me, opponent, games, SEED_BASE + 1000 * i)
        by_opponent[name] = statistics.fmean(points for points, _ in played)
    return {"overall": statistics.fmean(by_opponent.values()), **by_opponent}


def build_report(name: str, games: int = GAMES_PER_OPPONENT) -> dict[str, Any]:
    registry, metrics, artifacts = TelemetryStores.open()
    field_ = field_factories()
    rows = []
    for run in sorted(experiment_runs(registry, name), key=lambda r: (r.config["arm"], r.config["rng_seed"])):
        if run.status != "completed":
            continue
        history = metrics.history(run.run_id)
        snapshot = artifacts.get_program(history[-1].champion_ref).decode("utf-8")
        cost = (run.summary or {}).get("cost", {})
        rows.append(
            {
                "run_id": run.run_id,
                "arm": run.config["arm"],
                "rng_seed": run.config["rng_seed"],
                "points": score_run(snapshot, field_, games),
                "games": cost.get("episodes"),
                "active_s": cost.get("active_s"),
                "curve": [[h.generation, h.held_out_score] for h in history if h.held_out_score is not None],
            }
        )
        print(f"scored {run.config['arm']} seed {run.config['rng_seed']}: {rows[-1]['points']}", flush=True)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    arms = {}
    for arm, mine in by_arm.items():
        overall = [row["points"]["overall"] for row in mine]
        entry: dict[str, Any] = {
            "n": len(mine),
            "points_per_seed": overall,
            "points": _stats(overall),
            "by_opponent": {name: round(statistics.fmean(row["points"][name] for row in mine), 3) for name in field_},
            "active_s": _stats([row["active_s"] for row in mine if row["active_s"] is not None]),
        }
        baseline = ARMS[arm].baseline if arm in ARMS else None
        reference = {row["rng_seed"]: row["points"]["overall"] for row in by_arm.get(baseline or "", [])}
        paired = [
            (row["points"]["overall"], reference[row["rng_seed"]]) for row in mine if row["rng_seed"] in reference
        ]
        if len(paired) >= 2:
            a, b = zip(*paired, strict=True)
            entry["vs_baseline"] = {
                "arm": baseline,
                "mean_difference": round(statistics.fmean(x - y for x, y in paired), 3),
                "paired_p": round(paired_permutation_p(list(a), list(b)), 4),
                "pairs": len(paired),
            }
        arms[arm] = entry
    return {
        "name": name,
        "protocol": f"final network at {DEPTH}-ply vs {list(field_)}, {games} games each",
        "arms": arms,
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tracked Checkers self-play comparisons (docs/design/0010).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--arms", default=",".join(ARMS))
    run.add_argument("--seeds", default="0-4")
    report = sub.add_parser("report")
    report.add_argument("--name", required=True)
    report.add_argument("--games", type=int, default=GAMES_PER_OPPONENT, help="games per field opponent")
    args = parser.parse_args(argv)
    if args.command == "run":
        arms = args.arms.split(",")
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            sys.exit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
        run_experiment(args.name, arms, parse_seeds(args.seeds))
    else:
        result = build_report(args.name, args.games)
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        (EXPERIMENTS_DIR / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n## {result['name']} -- {result['protocol']}\n")
        for arm, a in result["arms"].items():
            vs = a.get("vs_baseline")
            versus = f"{vs['mean_difference']:+.3f} vs {vs['arm']} (p={vs['paired_p']:.3f})" if vs else "--"
            print(f"{arm:12} {a['points']['mean']:.3f} ± {a['points']['sd']:.3f}  {versus}  {a['by_opponent']}")


if __name__ == "__main__":
    main()
