"""Versus leaderboard evaluation (docs/design/0007, "Versus (Checkers)"): the two-player counterpart of
jobs/evaluate.py. Same `EvaluationRecord`s in the same store, so the game page's leaderboard components
work unchanged -- only what "a score" means differs.

Entrants for Checkers:
- every finished checkers run's final champion (a 32→H→1 position evaluator, jobs/checkers_neuro_run.py);
- the fixed baselines (random, first-legal, 1-ply and 2-ply material) -- always included, since a ranking
  says nothing without them.

Protocol `checkers.versus.v1`: a round robin. Every pair of entrants plays GAMES_PER_PAIR games, seats
alternating so first-move advantage cancels, each game seeded (random tie-breaks reproduce), capped at
MAX_PLIES. An entrant's **score is points per game** against every *other* entrant -- win 1, draw ½,
loss 0 -- with a 95% interval over those games; per-opponent win/draw/loss is kept beside it
(`metrics.versus`) for the head-to-head matrix. Rank by points per game; overlapping intervals show as
ties, exactly as for Snake.

Two properties a versus score has that Snake's doesn't, and the page says so:
- it is *relative to the field*: adding an entrant changes every score, so records are only comparable
  within one evaluation run. Re-run this job whenever entrants change (it replaces the old records);
- draws are real (40 moves without a capture), so a score of 0.5 is "even", not "no information".
Glicko-2 (docs/design/0007) is the planned rating; points per game is the honest first version.

Run with: uv run python jobs/evaluate_versus.py
"""

from __future__ import annotations

import itertools
import random
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

from costs import hardware_fingerprint
from evaluate import model_shape, training_cost
from evolve import NeatGenome, network_from_json, play_match
from evolve.networks import describe, parameter_count
from games import interfaces
from games.checkers import Checkers
from games.checkers_strategies import STRATEGIES, evaluator, graph_evaluator
from run_context import RUN_DATA_DIR, TelemetryStores
from telemetry import (
    EvaluationRecord,
    FileArtifactStore,
    FileMetricsStore,
    SqliteEvaluationStore,
    SqliteRunRegistry,
)

GAME = "checkers"
PROTOCOL = "checkers.versus.v1"
INTERFACE = "checkers/board32.v1+evaluate1ply.v1"
GAMES_PER_PAIR = 20  # 10 per seat
MAX_PLIES = 300
SEED_BASE = 30_000  # game seeds; disjoint from training's (jobs/checkers_neuro_run.py) and monitoring's

# Human names and one-line descriptions for the fixed baselines (the strategies themselves are Rust). The deeper
# material searches are the *fair* opponents for a trained evaluator that searches as deep: beating a shallower
# baseline than you search is a fact about search depth, not about what was learned.
BASELINES: dict[str, tuple[str, str]] = {
    "random": ("Random", "a uniformly random legal move"),
    "first-legal": ("First legal", "always the first legal move"),
    "material-1": ("Material, 1-ply", "the move leaving the best material balance"),
    "material-2": ("Material, 2-ply", "assumes the opponent's best reply; a tiny minimax"),
    "material-3": ("Material, 3-ply", "material, searched 3 plies with alpha-beta"),
    "material-4": ("Material, 4-ply", "material, searched 4 plies with alpha-beta"),
}

# A run shorter than this is a smoke test, not an entrant: its "champion" is the first generation's best.
MIN_GENERATIONS = 10

# (env, rng) -> Strategy
Factory = Callable[[Checkers, random.Random], Callable]


def baseline_entrants() -> list[dict[str, Any]]:
    return [
        {
            "entrant_id": f"baseline:{name}",
            "label": label,
            "factory": STRATEGIES[name],
            "model": description,
            "parameters": 0,
            "artifact_bytes": 0,
        }
        for name, (label, description) in BASELINES.items()
    ]


def champion_entrants(
    registry: SqliteRunRegistry, metrics: FileMetricsStore, artifacts: FileArtifactStore
) -> list[dict[str, Any]]:
    entrants = []
    for run in registry.list_runs():
        if run.config.get("game") != GAME or run.config.get("interface") != INTERFACE:
            continue
        # Only finished runs: a running or failed run's latest champion isn't its final one.
        if run.status != "completed" or run.config.get("experiment"):
            continue
        history = metrics.history(run.run_id)
        if len(history) < MIN_GENERATIONS:
            continue
        champion_ref = history[-1].champion_ref
        raw = artifacts.get_program(champion_ref)
        champion = network_from_json(raw.decode("utf-8"))
        network = describe(champion)
        selection = run.config.get("selection")
        depth = int(run.config.get("search_depth", 1))
        neat = isinstance(champion, NeatGenome)
        kind = "NEAT" if neat else "Neuroevolution"
        searching = f"{depth}-ply search" if depth > 1 else "one ply"
        entrants.append(
            {
                "entrant_id": f"run:{run.run_id}",
                "label": f"{kind} {network}"
                + (f" · {selection}" if selection else "")
                + (f" · {depth}-ply" if depth > 1 else ""),
                "factory": graph_evaluator(champion.graph_encoding(), depth)
                if neat
                else evaluator(champion.weights, champion.layer_sizes, depth),
                "run": run,
                "champion_ref": champion_ref,
                "model": f"{'evolved graph' if neat else 'MLP'} {network}, tanh: a position evaluator, {searching}",
                "search_depth": depth,
                "parameters": parameter_count(champion),
                "artifact_bytes": len(raw),
                "shape": model_shape(champion, kind, selection),
                "note": run.config.get("note"),
            }
        )
    return entrants


# --- Measurement -------------------------------------------------------------------------------------


def play_pairing(a: Factory, b: Factory, games: int, seed_base: int) -> list[tuple[float, int]]:
    """`games` games between `a` and `b`, seats alternating: (a's points, plies) per game."""
    results = []
    for game in range(games):
        seed = seed_base + game
        env = Checkers()
        seat_a = game % 2
        strategies = {
            seat_a: a(env, random.Random(seed)),
            1 - seat_a: b(env, random.Random(seed + 7919)),
        }
        match = play_match(env, strategies, max_moves=MAX_PLIES)
        points = 0.5 if match.winner is None else 1.0 if match.winner == seat_a else 0.0
        results.append((points, match.moves_played))
    return results


def round_robin(
    entrants: list[dict[str, Any]], games: int = GAMES_PER_PAIR
) -> dict[str, dict[str, list[tuple[float, int]]]]:
    """results[a][b] = a's (points, plies) games against b (b's are the mirror image)."""
    results: dict[str, dict[str, list[tuple[float, int]]]] = {e["entrant_id"]: {} for e in entrants}
    for index, (a, b) in enumerate(itertools.combinations(entrants, 2)):
        played = play_pairing(a["factory"], b["factory"], games, SEED_BASE + index * games)
        results[a["entrant_id"]][b["entrant_id"]] = played
        results[b["entrant_id"]][a["entrant_id"]] = [(1.0 - points, plies) for points, plies in played]
    return results


def quality_of(games: list[tuple[float, int]]) -> dict[str, Any]:
    points = [p for p, _ in games]
    n = len(points)
    stdev = statistics.stdev(points) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(statistics.fmean(points), 4),
        "ci95": round(1.96 * stdev / n**0.5, 4) if n > 1 else 0.0,
        "median": statistics.median(points),
        "min": min(points),
        "max": max(points),
        "zero_rate": round(sum(1 for p in points if p == 0.0) / n, 4),  # the loss rate
        "mean_steps": round(statistics.fmean(plies for _, plies in games), 2),
        "train_mean": None,  # no training-seed benchmark to compare with: the opponents *are* the benchmark
        "generalization_gap": None,
        "scores": points,
    }


def versus_of(by_opponent: dict[str, list[tuple[float, int]]]) -> dict[str, Any]:
    def wdl(games: list[tuple[float, int]]) -> dict[str, int]:
        return {
            "wins": sum(1 for p, _ in games if p == 1.0),
            "draws": sum(1 for p, _ in games if p == 0.5),
            "losses": sum(1 for p, _ in games if p == 0.0),
        }

    everything = [g for games in by_opponent.values() for g in games]
    return {
        **wdl(everything),
        "games_per_pair": len(next(iter(by_opponent.values()))),
        "by_opponent": {k: wdl(v) for k, v in by_opponent.items()},
    }


def measure_inference(factory: Factory, positions: int = 200, repeats: int = 3) -> dict[str, Any]:
    """Median-of-repeats time per decision on real mid-game positions. The observation is encoded inside
    the native call, so there is no separate encode step to time."""
    rng = random.Random(0)
    envs = []
    env = Checkers()
    env.reset()
    for _ in range(positions):
        if env.winner() is not None or not env.legal_moves():
            env.reset()
        envs.append(_copy(env))
        env.step(rng.choice(env.legal_moves()))

    samples = []
    for _ in range(repeats):
        bound = [(e, factory(e, random.Random(1))) for e in envs]  # building a strategy isn't a decision
        started = time.perf_counter()
        for e, strategy in bound:
            strategy(None, e.legal_moves())
        samples.append((time.perf_counter() - started) / len(bound) * 1e6)
    decide_us = statistics.median(samples)
    return {"encode_us": 0.0, "decide_us": round(decide_us, 3), "total_us": round(decide_us, 3)}


def _copy(env: Checkers) -> Checkers:
    import copy

    return copy.deepcopy(env)


def evaluate_all(
    entrants: list[dict[str, Any]],
    metrics: FileMetricsStore | None,
    hardware: dict[str, Any],
    games: int = GAMES_PER_PAIR,
) -> list[EvaluationRecord]:
    results = round_robin(entrants, games)
    interface = interfaces.get(INTERFACE)
    records = []
    for entrant in entrants:
        by_opponent = results[entrant["entrant_id"]]
        all_games = [g for played in by_opponent.values() for g in played]
        inference = measure_inference(entrant["factory"])
        inference["parameters"] = entrant["parameters"]
        inference["artifact_bytes"] = entrant["artifact_bytes"]
        run = entrant.get("run")
        records.append(
            EvaluationRecord(
                game=GAME,
                protocol=PROTOCOL,
                entrant_id=entrant["entrant_id"],
                entrant_kind="champion" if run else "baseline",
                label=entrant["label"],
                interface=INTERFACE,
                run_id=run.run_id if run else None,
                champion_ref=entrant.get("champion_ref"),
                created_at=time.time(),
                metrics={
                    "quality": quality_of(all_games),
                    "versus": versus_of(by_opponent),
                    "inference": inference,
                    "training": training_cost(run, metrics) if run and metrics else {"measured": True, "none": True},
                    "model": {
                        "description": entrant["model"],
                        "search_depth": entrant.get("search_depth", 1),
                        "shape": entrant.get("shape"),
                        "observer_level": interface.observer.level,
                        "note": entrant.get("note"),
                    },
                    "protocol": {
                        "held_out_seeds": [SEED_BASE, SEED_BASE + games * len(entrants) * (len(entrants) - 1) // 2 - 1],
                        "episodes": len(all_games),
                        "max_steps": MAX_PLIES,
                        "board": {"width": 8, "height": 8},
                        "metric": "points per game (win 1, draw ½, loss 0) against every other entrant, both seats",
                    },
                },
                hardware=hardware,
            )
        )
    return records


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # labels contain arrows; the Windows console default can't print them
    registry, metrics, artifacts = TelemetryStores.open()
    store = SqliteEvaluationStore(RUN_DATA_DIR / "evaluations.db")

    entrants = [*baseline_entrants(), *champion_entrants(registry, metrics, artifacts)]
    records = evaluate_all(entrants, metrics, hardware_fingerprint())
    for record in sorted(records, key=lambda r: -r.metrics["quality"]["mean"]):
        store.put(record)
        q, v = record.metrics["quality"], record.metrics["versus"]
        print(
            f"{record.label:<44} {q['mean']:.3f} ±{q['ci95']:.3f}  "
            f"{v['wins']}W {v['draws']}D {v['losses']}L over {q['n']} games  "
            f"{record.metrics['inference']['total_us']:>8.1f} µs/decision"
        )


if __name__ == "__main__":
    main()
