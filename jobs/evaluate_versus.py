"""Versus leaderboard evaluation (docs/design/0007, "Versus (Checkers)"): the two-player counterpart of
jobs/evaluate.py. Same `EvaluationRecord`s in the same store, so the game page's leaderboard components
work unchanged -- only what "a score" means differs.

Entrants for Checkers:
- every finished checkers run's final champion (a 32→H→1 position evaluator, jobs/checkers_neuro_run.py);
- the fixed baselines (random, first-legal, 1-ply and 2-ply material) -- always included, since a ranking
  says nothing without them.

Protocol `checkers.versus.v2` (docs/design/0013): a round robin over the **ballot** (games.checkers_openings: 174
level three-ply openings). Every pair of entrants plays OPENINGS_PER_PAIR openings, each as a **game pair** -- once
from each seat -- so opening luck cancels; each pairing takes the next openings in the ballot, so the round robin
covers all of it. Games are seeded (random tie-breaks reproduce) and capped at MAX_PLIES.

An entrant's **score is an Elo rating**: Bradley-Terry fitted to every game of the round robin (arena.versus_stats),
anchored so Random is 0, with a bootstrap 95% interval over game pairs. `quality.mean` holds the rating, so every
leaderboard component ranks by it unchanged; points per game, the pentanomial pair counts and the per-opponent
win/draw/loss (the head-to-head matrix) are in `metrics.versus`. Overlapping intervals show as ties.

Why a rating rather than points per game (v1): points against the field saturate -- a leader that beats a field of
much weaker entrants scores ~0.94 whatever it is -- while a rating is set by the games it played against opponents
near its strength. It is still relative to the field (re-run this job whenever entrants change; it replaces the old
records), and draws are real (40 moves without a capture), counting half a win each way.

Run with: uv run python jobs/evaluate_versus.py
"""

from __future__ import annotations

import itertools
import random
import statistics
import sys
import time
from typing import Any

from arena.checkers import (
    ANCHOR,
    GAME,
    INTERFACE,
    MAX_PLIES,
    OPENINGS_PER_PAIR,
    PROTOCOL,
    VERSUS_SEED_BASE,
    Factory,
    openings_for,
    pair_points,
    play_pairing,
)
from arena.costs import hardware_fingerprint
from arena.versus_stats import pentanomial, rate
from evaluate import model_shape, training_cost
from evolve import NeatGenome, network_from_json
from evolve.networks import describe, parameter_count
from games import interfaces
from games.checkers import Checkers
from games.checkers_openings import ballot
from games.checkers_strategies import STRATEGIES, evaluator, graph_evaluator
from telemetry import (
    EvaluationRecord,
    FileArtifactStore,
    FileMetricsStore,
    SqliteRunRegistry,
    open_stores,
)

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
        # how it was trained: evolved (NEAT or a fixed network), or learned by self-play (docs/design/0010 Phase 4)
        # (TD-Leaf is TD(λ) trained through its own search: `search_depth` > 1 among the self-play params)
        # (an AlphaZero run's champion is its value head: the trunk and value unit, docs/design/0017)
        leaf = int((run.config.get("params") or {}).get("search_depth", 1)) > 1
        representation = run.config.get("representation")
        if neat:
            kind = "NEAT"
        elif representation == "td_lambda":
            kind = "TD-Leaf(λ) self-play" if leaf else "TD(λ) self-play"
        elif representation == "alphazero":
            kind = "AlphaZero self-play"
        else:
            kind = "Neuroevolution"
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
                "model": f"{'evolved graph' if neat else 'MLP'} {network}, tanh: a position evaluator, {searching}"
                + (f", trained by {kind}" if "self-play" in kind else ""),
                "search_depth": depth,
                "parameters": parameter_count(champion),
                "artifact_bytes": len(raw),
                "shape": model_shape(champion, kind, selection),
                "note": run.config.get("note"),
            }
        )
    return entrants


# --- Measurement -------------------------------------------------------------------------------------


def round_robin(
    entrants: list[dict[str, Any]], per_pair: int = OPENINGS_PER_PAIR
) -> dict[str, dict[str, list[tuple[float, int]]]]:
    """results[a][b] = a's (points, plies) games against b, in game pairs (b's are the mirror image)."""
    results: dict[str, dict[str, list[tuple[float, int]]]] = {e["entrant_id"]: {} for e in entrants}
    for index, (a, b) in enumerate(itertools.combinations(entrants, 2)):
        played = play_pairing(
            a["factory"], b["factory"], openings_for(index, per_pair), VERSUS_SEED_BASE + index * 2 * per_pair
        )
        results[a["entrant_id"]][b["entrant_id"]] = played
        results[b["entrant_id"]][a["entrant_id"]] = [(1.0 - points, plies) for points, plies in played]
    return results


def ratings(results: dict[str, dict[str, list[tuple[float, int]]]], replicates: int = 200) -> dict[str, dict]:
    """Bradley-Terry Elo for every entrant (ANCHOR = 0) with bootstrap intervals over game pairs."""
    pairings = {}
    for a, by_opponent in results.items():
        for b, games in by_opponent.items():
            if (b, a) not in pairings:
                pairings[(a, b)] = pair_points(games)
    return rate(pairings, anchor=ANCHOR, replicates=replicates)


def quality_of(games: list[tuple[float, int]], rating: dict[str, float]) -> dict[str, Any]:
    """The score block every leaderboard reads: here the Elo rating (`mean`), its interval's half-width (`ci95`) and
    ends (`min`/`max`); `scores` stays the per-game points."""
    points = [p for p, _ in games]
    n = len(points)
    return {
        "n": n,
        "mean": rating["elo"],
        "ci95": round((rating["hi"] - rating["lo"]) / 2, 1),
        "median": rating["elo"],
        "min": rating["lo"],
        "max": rating["hi"],
        "zero_rate": round(sum(1 for p in points if p == 0.0) / n, 4),  # the loss rate
        "mean_steps": round(statistics.fmean(plies for _, plies in games), 2),
        "train_mean": None,  # no training-seed benchmark to compare with: the opponents *are* the benchmark
        "generalization_gap": None,
        "scores": points,
    }


def versus_of(by_opponent: dict[str, list[tuple[float, int]]], rating: dict[str, float]) -> dict[str, Any]:
    def wdl(games: list[tuple[float, int]]) -> dict[str, int]:
        return {
            "wins": sum(1 for p, _ in games if p == 1.0),
            "draws": sum(1 for p, _ in games if p == 0.5),
            "losses": sum(1 for p, _ in games if p == 0.0),
        }

    everything = [g for games in by_opponent.values() for g in games]
    return {
        **wdl(everything),
        "points": round(statistics.fmean(p for p, _ in everything), 4),
        "pentanomial": pentanomial([p for games in by_opponent.values() for p in pair_points(games)]),
        "rating": {**rating, "anchor": ANCHOR, "scale": "Elo (Bradley-Terry)"},
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
    per_pair: int = OPENINGS_PER_PAIR,
    replicates: int = 200,
) -> list[EvaluationRecord]:
    results = round_robin(entrants, per_pair)
    rated = ratings(results, replicates)
    interface = interfaces.get(INTERFACE)
    pairings = len(entrants) * (len(entrants) - 1) // 2
    records = []
    for entrant in entrants:
        by_opponent = results[entrant["entrant_id"]]
        all_games = [g for played in by_opponent.values() for g in played]
        rating = rated[entrant["entrant_id"]]
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
                    "quality": quality_of(all_games, rating),
                    "versus": versus_of(by_opponent, rating),
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
                        "held_out_seeds": [VERSUS_SEED_BASE, VERSUS_SEED_BASE + 2 * per_pair * pairings - 1],
                        "episodes": len(all_games),
                        "max_steps": MAX_PLIES,
                        "board": {"width": 8, "height": 8},
                        "openings": {"ballot": len(ballot()), "per_pair": per_pair},
                        "metric": "Elo (Bradley-Terry over every game, Random = 0), from game pairs on ballot openings "
                        "against every other entrant",
                    },
                },
                hardware=hardware,
            )
        )
    return records


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # labels contain arrows; the Windows console default can't print them
    stores = open_stores()
    registry, metrics, artifacts = stores.registry, stores.metrics, stores.artifacts
    store = stores.evaluations

    entrants = [*baseline_entrants(), *champion_entrants(registry, metrics, artifacts)]
    records = evaluate_all(entrants, metrics, hardware_fingerprint())
    for record in sorted(records, key=lambda r: -r.metrics["quality"]["mean"]):
        store.put(record)
        q, v = record.metrics["quality"], record.metrics["versus"]
        print(
            f"{record.label:<52} Elo {q['mean']:>6.0f} [{q['min']:.0f}, {q['max']:.0f}]  {v['points']:.3f} pts/game  "
            f"{v['wins']}W {v['draws']}D {v['losses']}L over {q['n']} games  "
            f"{record.metrics['inference']['total_us']:>8.1f} µs/decision"
        )
    # A versus score is relative to the field, so a stale record (an entrant that no longer qualifies) is wrong twice.
    for entrant_id in store.prune(GAME, PROTOCOL, {r.entrant_id for r in records}):
        print(f"removed {entrant_id}: no longer an entrant")


if __name__ == "__main__":
    main()
