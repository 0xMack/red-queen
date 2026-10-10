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

`selfplay-v4` compares opponent regimes (2 x 64, 200k games, `g-pool` the baseline): `g-sp` (itself only), `g-league`
(every past self, one every 5k games), `g-pfsp` (the last 10, prioritized toward those it doesn't beat), `g-league-pfsp`,
and `g-pool-80` (80% of games against the pool). Report it with `h2h --full`. `g-pbt` is population-based training
(jobs/checkers_pbt_run.py: 8 members, the settings evolving), `g-rs` the same population with nothing copied.

`selfplay-v5` asks whether TD-Leaf should train from the start (docs/design/0015): `s-leaf2` / `s-leaf2-40k` (equal games /
equal compute against plain TD, `s-td`), and `s-leaf2-80k` against the fine-tuning recipe `s-td-ft2` (TD, then TD-Leaf)
at about equal total compute. `s-td-ft2` continues each seed's `s-td`, so run `s-td` first.

`selfplay-v6` asks what limits the full recipe (plain TD, then 40k games of TD-Leaf at 2 plies): the length of the TD
phase, or the network. `l-1m-ft2` fine-tunes `selfplay-v2`'s 1M-game 2 x 64 networks (against `selfplay-v5`'s
`s-td-ft2`, the same fine-tune after 200k); `l-3x64-1m` and `l-2x128-1m` are deeper and wider networks trained 1M games
of plain TD (against `pool-2x64-1m`), and `l-3x64-ft2` / `l-2x128-ft2` their fine-tunes (against `l-1m-ft2`). A
baseline in another experiment is an `(experiment, arm)` pair.

`selfplay-v7` (prepared) keeps scaling width: `w-2x256-1m` and its fine-tune `w-2x256-ft2`, and `w-2x64-2700k`, a 2 x 64
network given 2 x 128's compute in games, to tell capacity from training time. `selfplay-v8` (prepared) is AlphaZero-style
self-play (`algorithm="alphazero"`, docs/design/0017): `z-az-ft` / `z-az-ft-16k` fine-tune the 1M-game 2 x 64 networks
with search targets (against `l-1m-ft2`, at equal games / about equal compute) and `z-az-200k` learns from scratch
(against `s-leaf2`). An AlphaZero arm's network is its value head, so `h2h` compares evaluators under the same search.

  uv run python jobs/checkers_selfplay_experiment.py run    --name NAME --arms sp,sp-pool --seeds 0-4
  uv run python jobs/checkers_selfplay_experiment.py report --name NAME [--games 20]
  uv run python jobs/checkers_selfplay_experiment.py h2h    --name NAME [--arms ft-leaf3] [--elo1 50] [--full]

`h2h` is the comparison that doesn't run out of opponents (docs/design/0013): each arm's final network plays its
baseline arm's network from the same rng seed, head to head, both searching `DEPTH` plies, in game pairs over the
ballot -- every seed's couple plays each opening before the next -- feeding one SPRT (jobs/checkers_sprt.py) that
stops once the arm is shown `elo1` stronger (H1) or no stronger (H0). `--full` plays the whole ballot regardless, for
the tightest estimate. Reported per arm: the verdict, the pooled Elo difference with its interval, and each seed's.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

import checkers_pbt_run
import checkers_selfplay_run
from arena.checkers import REPORT_SEED_BASE, play_pairing
from arena.versus_stats import Sprt, pair_elo
from checkers_sprt import run_factory, sequential_match
from evolve import WeightVector, network_from_json
from games.checkers_openings import ballot
from games.checkers_strategies import STRATEGIES, evaluator
from jobcore import experiments_dir, open_sink
from rl_experiment import paired_permutation_p
from snake_experiment import _stats, experiment_runs, parse_seeds

DEPTH = 3
GAMES_PER_OPPONENT = 20  # games (so GAMES_PER_OPPONENT / 2 ballot openings, a game pair each) per field opponent
FIELD = ("material-2", "material-3", "material-4")
# The versus leaderboard's best evolved evaluator (0.757 points per game there): the benchmark from the other paradigm.
EVOLVED_RUN = "802e1c7624ae402595d5384aaa1da2e8"


@dataclass(frozen=True)
class Arm:
    params: dict[str, float] = field(default_factory=dict)
    games: int = 200_000
    baseline: str | tuple[str, str] | None = "sp"  # an arm of this experiment, or (experiment, arm)
    # Continue from the final network of this (experiment, arm)'s run with the same rng seed, instead of a fresh one.
    init: tuple[str, str] | None = None
    # A population (jobs/checkers_pbt_run.py) of this many members instead of one learner: `games` per member.
    population: int = 0
    exploit: bool = True
    algorithm: str = "td_lambda"  # or "alphazero" (jobs/checkers_selfplay_run.py)


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
    # selfplay-v4, opponent regimes (2 x 64, 200k games; measured with `h2h`, docs/design/0013): who does the network
    # play? Itself only, the last 10 selves (the recipe so far), every past self (a league: fictitious self-play), the
    # past selves it still fails to beat (prioritized fictitious self-play, AlphaStar's PFSP), or mostly the pool.
    "g-pool": Arm(BIG, baseline=None),
    "g-sp": Arm({"hidden": 64, "hidden_layers": 2}, baseline="g-pool"),
    "g-league": Arm({**BIG, "pool_size": 40}, baseline="g-pool"),
    "g-pfsp": Arm({**BIG, "pfsp": 2}, baseline="g-pool"),
    "g-league-pfsp": Arm({**BIG, "pool_size": 40, "pfsp": 2}, baseline="g-league"),
    "g-pool-80": Arm({**BIG, "pool_fraction": 0.8}, baseline="g-pool"),
    # Population-based training (8 members, 200k games each, settings sampled then evolved) against the same population
    # with nothing copied -- random search over the same settings, equal compute, best member kept.
    "g-rs": Arm(population=8, exploit=False, baseline="g-pool"),
    "g-pbt": Arm(population=8, baseline="g-rs"),
    # selfplay-v5: TD-Leaf from the start, not just as a finishing step (2 x 64, the pool). TD-Leaf at 2 plies costs about
    # 5x a plain-TD game, so each from-scratch arm is matched to a TD arm on games *or* on compute:
    "s-td": Arm(BIG, baseline=None),  # plain TD, 200k games (the selfplay-v4 recipe)
    # TD-Leaf from scratch, the same 200k games (~5x compute)
    "s-leaf2": Arm({**BIG, "search_depth": 2}, baseline="s-td"),
    "s-leaf2-40k": Arm({**BIG, "search_depth": 2}, games=40_000, baseline="s-td"),  # ... the same compute as s-td
    # the fine-tuning recipe (plain TD, then TD-Leaf), and from-scratch TD-Leaf at about the same total compute
    "s-td-ft2": Arm({**BIG, "search_depth": 2}, games=40_000, baseline="s-td", init=("selfplay-v5", "s-td")),
    "s-leaf2-80k": Arm({**BIG, "search_depth": 2}, games=80_000, baseline="s-td-ft2"),
    # selfplay-v6: what limits the recipe -- the TD phase's length, or the network? (each fine-tune: 40k games, 2 plies)
    "l-1m-ft2": Arm(
        {**BIG, "search_depth": 2},
        games=40_000,
        baseline=("selfplay-v5", "s-td-ft2"),
        init=("selfplay-v2", "pool-2x64-1m"),
    ),
    # 32 -> 64 -> 64 -> 64 -> 1 (10.5k weights) and 32 -> 128 -> 128 -> 1 (20.7k), against 2 x 64 (6.3k)
    "l-3x64-1m": Arm({**BIG, "hidden_layers": 3}, games=1_000_000, baseline=("selfplay-v2", "pool-2x64-1m")),
    "l-2x128-1m": Arm({**BIG, "hidden": 128}, games=1_000_000, baseline=("selfplay-v2", "pool-2x64-1m")),
    "l-3x64-ft2": Arm(
        {**BIG, "hidden_layers": 3, "search_depth": 2},
        games=40_000,
        baseline="l-1m-ft2",
        init=("selfplay-v6", "l-3x64-1m"),
    ),
    "l-2x128-ft2": Arm(
        {**BIG, "hidden": 128, "search_depth": 2},
        games=40_000,
        baseline="l-1m-ft2",
        init=("selfplay-v6", "l-2x128-1m"),
    ),
    # selfplay-v7 (prepared, not yet run; docs/design/0016's "next"): does width keep paying, and is 2 x 128's gain
    # capacity or just compute? 32 -> 256 -> 256 -> 1 (74k weights, ~3.6x 2 x 128's) against 2 x 128, a 2 x 64 network
    # given 2 x 128's compute in games (2.7M) against 2 x 128, and 2 x 256's fine-tune against 2 x 128's.
    "w-2x256-1m": Arm({**BIG, "hidden": 256}, games=1_000_000, baseline=("selfplay-v6", "l-2x128-1m")),
    "w-2x64-2700k": Arm(BIG, games=2_700_000, baseline=("selfplay-v6", "l-2x128-1m")),
    "w-2x256-ft2": Arm(
        {**BIG, "hidden": 256, "search_depth": 2},
        games=40_000,
        baseline=("selfplay-v6", "l-2x128-ft2"),
        init=("selfplay-v7", "w-2x256-1m"),
    ),
    # selfplay-v8 (prepared, not yet run; docs/design/0017): AlphaZero-style search targets. Each value head is compared
    # with the TD-trained network it would replace, both searched by alpha-beta at DEPTH plies -- the question is
    # whether a value learned from MCTS self-play is a better *evaluator*. `z-az-ft` fine-tunes the same 1M-game 2 x 64
    # TD networks TD-Leaf fine-tuned in `l-1m-ft2` (equal games: 40k; ~2.4x its compute at 50 simulations), and
    # `z-az-ft-16k` matches that compute instead. `z-az-200k` learns from scratch, against TD-Leaf from scratch for the
    # same 200k games (`s-leaf2`; ~1.75x its compute).
    "z-az-ft": Arm(
        {"hidden": 64, "hidden_layers": 2, "simulations": 50},
        games=40_000,
        baseline=("selfplay-v6", "l-1m-ft2"),
        init=("selfplay-v2", "pool-2x64-1m"),
        algorithm="alphazero",
    ),
    "z-az-ft-16k": Arm(
        {"hidden": 64, "hidden_layers": 2, "simulations": 50},
        games=16_000,
        baseline=("selfplay-v6", "l-1m-ft2"),
        init=("selfplay-v2", "pool-2x64-1m"),
        algorithm="alphazero",
    ),
    "z-az-200k": Arm(
        {"hidden": 64, "hidden_layers": 2, "simulations": 50},
        baseline=("selfplay-v5", "s-leaf2"),
        algorithm="alphazero",
    ),
}
GAMES_PER_ITERATION = 1000


def completed_by_seed(registry: Any, experiment: str, arm: str) -> dict[int, Any]:
    """An arm's completed runs, by rng seed."""
    return {
        r.config["rng_seed"]: r
        for r in experiment_runs(registry, experiment)
        if r.config.get("arm") == arm and r.status == "completed"
    }


def resolve_baseline(name: str, arm: str) -> tuple[str, str] | None:
    """An arm's baseline as (experiment, arm): a bare arm name is in this experiment."""
    baseline = ARMS[arm].baseline if arm in ARMS else None
    if baseline is None or isinstance(baseline, tuple):
        return baseline
    return (name, baseline)


def run_experiment(name: str, arms: list[str], seeds: list[int]) -> None:
    registry = open_sink().registry
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
                parent = completed_by_seed(registry, *arm.init).get(seed)
                if parent is None:
                    sys.exit(f"{arm_name} seed {seed} continues {arm.init}, which has no completed run for that seed")
                init_run = parent.run_id
            print(f"=== {name} · {arm_name} · seed {seed} ({arm.games:,} games)", flush=True)
            if arm.population:
                checkers_pbt_run.main(
                    members=arm.population,
                    games=arm.games,
                    depth=DEPTH,
                    exploit=arm.exploit,
                    rng_seed=seed,
                    tags={"experiment": name, "arm": arm_name},
                )
                continue
            iterations = max(1, arm.games // GAMES_PER_ITERATION)
            checkers_selfplay_run.main(
                iterations=iterations,
                games_per_iteration=GAMES_PER_ITERATION,
                depth=DEPTH,
                params=dict(arm.params),
                rng_seed=seed,
                held_out_every=max(1, iterations // 10),
                tags={"experiment": name, "arm": arm_name},
                init_run=init_run,
                algorithm=arm.algorithm,
            )


def field_factories() -> dict[str, Any]:
    """The fixed opponents: material search, and the best evolved evaluator (if this machine has its run)."""
    factories = {name: STRATEGIES[name] for name in FIELD}
    sink = open_sink()
    registry, metrics, artifacts = sink.registry, sink.metrics, sink.artifacts
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
    openings = list(ballot())
    for i, (name, opponent) in enumerate(field_.items()):
        chosen = [openings[(i * 37 + k) % len(openings)] for k in range(max(1, games // 2))]
        played = play_pairing(me, opponent, chosen, REPORT_SEED_BASE + 1000 * i)
        by_opponent[name] = statistics.fmean(points for points, _ in played)
    return {"overall": statistics.fmean(by_opponent.values()), **by_opponent}


def build_report(name: str, games: int = GAMES_PER_OPPONENT) -> dict[str, Any]:
    sink = open_sink()
    registry, metrics, artifacts = sink.registry, sink.metrics, sink.artifacts
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
        resolved = resolve_baseline(name, arm)
        # the field score is only computed for this experiment's runs
        baseline = resolved[1] if resolved and resolved[0] == name else None
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


def across_seeds(per_seed: dict[int, dict[str, float]]) -> dict[str, Any]:
    """The regime-level view of a head-to-head. The pooled interval counts every game pair, so it answers "are these
    networks stronger than those" -- but a regime is a distribution over training seeds, and seeds differ. Here each
    seed's Elo difference is one sample: their mean and an exact sign-flip permutation test against 0 (with 5 seeds the
    smallest possible p is 0.0625, whatever the games say)."""
    elos = [e["elo"] for e in per_seed.values()]
    return {
        "mean": round(statistics.fmean(elos), 1),
        "sd": round(statistics.stdev(elos), 1) if len(elos) > 1 else 0.0,
        "positive": sum(1 for e in elos if e > 0),
        "seeds": len(elos),
        "sign_flip_p": round(paired_permutation_p(elos, [0.0] * len(elos)), 4) if len(elos) > 1 else None,
    }


def build_h2h(name: str, arms: list[str] | None = None, elo1: float = 50.0, full: bool = False) -> dict[str, Any]:
    """Each arm (with a baseline) against its baseline arm, seed for seed, head to head (see the module docs)."""
    sink = open_sink()
    registry, metrics, artifacts = sink.registry, sink.metrics, sink.artifacts
    runs = [r for r in experiment_runs(registry, name) if r.status == "completed"]
    by_arm: dict[str, dict[int, Any]] = {}
    for run in runs:
        by_arm.setdefault(run.config["arm"], {})[run.config["rng_seed"]] = run
    out = {}
    for arm in arms or sorted(by_arm):
        resolved = resolve_baseline(name, arm)
        if resolved is None or arm not in by_arm:
            continue
        theirs = by_arm.get(resolved[1], {}) if resolved[0] == name else completed_by_seed(registry, *resolved)
        baseline = resolved[1] if resolved[0] == name else f"{resolved[0]}/{resolved[1]}"
        seeds = sorted(set(by_arm[arm]) & set(theirs))
        if not seeds:
            continue
        couples = [
            (
                run_factory(by_arm[arm][s], metrics, artifacts, DEPTH),
                run_factory(theirs[s], metrics, artifacts, DEPTH),
            )
            for s in seeds
        ]
        # --full: bounds no finite LLR reaches, so the whole ballot is played
        test = Sprt(elo0=0.0, elo1=elo1, alpha=1e-300, beta=1e-300) if full else Sprt(elo0=0.0, elo1=elo1)
        print(f"=== {arm} vs {baseline}: {len(seeds)} seeds, {len(ballot())} openings available", flush=True)
        result = sequential_match(couples, test)
        per_seed = {seed: pair_elo(pairs) for seed, pairs in zip(seeds, result.pop("per_couple"), strict=True)}
        if full:
            result["verdict"] = "full ballot"
            result["alpha"] = result["beta"] = None
        out[arm] = {
            "baseline": baseline,
            "seeds": seeds,
            **result,
            "per_seed": per_seed,
            "across_seeds": across_seeds(per_seed),
        }
        print(
            f"{arm} vs {baseline}: {result['verdict']} after {result['openings_played']} openings x {len(seeds)} seeds "
            f"-- Elo {result['elo']:+.0f} [{result['lo']:+.0f}, {result['hi']:+.0f}]  "
            + "  ".join(f"s{s} {e['elo']:+.0f}" for s, e in per_seed.items()),
            flush=True,
        )
    return {
        "name": name,
        "protocol": f"head to head at {DEPTH}-ply, game pairs over the {len(ballot())}-opening ballot, "
        + ("the whole ballot" if full else f"SPRT elo0 0, elo1 {elo1}, alpha = beta = 0.05"),
        "arms": out,
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
    h2h = sub.add_parser("h2h")
    h2h.add_argument("--name", required=True)
    h2h.add_argument("--arms", default=None, help="comma list (default: every arm with a baseline)")
    h2h.add_argument("--elo1", type=float, default=50.0)
    h2h.add_argument("--full", action="store_true", help="play the whole ballot instead of stopping early")
    args = parser.parse_args(argv)
    if args.command == "h2h":
        result = build_h2h(args.name, args.arms.split(",") if args.arms else None, args.elo1, args.full)
        experiments_dir().mkdir(parents=True, exist_ok=True)
        suffix = "-h2h-full" if args.full else "-h2h"
        (experiments_dir() / f"{args.name}{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if args.command == "run":
        arms = args.arms.split(",")
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            sys.exit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
        run_experiment(args.name, arms, parse_seeds(args.seeds))
    else:
        result = build_report(args.name, args.games)
        experiments_dir().mkdir(parents=True, exist_ok=True)
        (experiments_dir() / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n## {result['name']} -- {result['protocol']}\n")
        for arm, a in result["arms"].items():
            vs = a.get("vs_baseline")
            versus = f"{vs['mean_difference']:+.3f} vs {vs['arm']} (p={vs['paired_p']:.3f})" if vs else "--"
            print(f"{arm:12} {a['points']['mean']:.3f} ± {a['points']['sd']:.3f}  {versus}  {a['by_opponent']}")


if __name__ == "__main__":
    main()
