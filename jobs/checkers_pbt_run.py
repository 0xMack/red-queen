"""Population-based training for Checkers self-play (docs/design/0014): evolution choosing the settings while TD(λ)
trains the weights.

A population of self-play learners (rl.CheckersSelfPlay, 2 x 64 tanh evaluators) trains side by side, each with its own
hyperparameters -- learning rate, λ, how often it plays its pool of past selves, whether it prioritizes the ones it
can't beat -- sampled at the start. Every `interval` games per member the members play a round robin among themselves
(game pairs over ballot openings, searching `depth` plies, docs/design/0013), and with `exploit`:

- the bottom quarter copy the weights of a random member of the top quarter (**exploit**) and take its settings,
- each copied setting is then nudged -- x0.8 or x1.25, λ and the pool fraction kept in range, the PFSP switch flipped
  one time in five (**explore**).

So settings that produce stronger players spread, and are tuned while they do: a schedule is discovered, not chosen
(Jaderberg et al. 2017, "Population Based Training of Neural Networks"). `exploit=False` is the control that uses the
same population, the same sampled settings and the same compute with nothing copied: random search, best member kept.

Recorded as one run: each generation's best/mean/worst are the round robin's points per game, the champion is the
round's best member (so the leaderboard and the Checkers page take it like any other self-play champion), and extras
carry the best member's settings and how many were replaced. The artifact `<run_id>-population` holds every member's
score and settings each generation and who copied whom (note: the settings logged for generation g are the ones the
members train with *next*, after that round's exploit/explore).

  uv run python jobs/checkers_pbt_run.py [--members 8] [--games 200000] [--interval 10000] [--no-exploit]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from typing import Any

import rl
from checkers_selfplay_run import INTERFACE, MAX_MOVES, MAX_MOVES_WITHOUT_CAPTURE, OPPONENTS
from checkers_training import MONITOR_GAMES, monitor_score
from costs import TrainingCostMeter
from evaluate_versus import pair_points, play_pairing
from evolve import WeightVector
from games.checkers_openings import ballot
from games.checkers_strategies import evaluator
from run_context import recorded_run
from telemetry import GenerationStats

BASE = {"hidden": 64, "hidden_layers": 2, "pool_every": 5000, "pool_size": 10}
SEED_BASE = 60_000  # round-robin game seeds; disjoint from every other measurement's
PAUSE_CHECK_GAMES = 1000  # a paused run stops within this many games of one member


def sample_settings(rng: random.Random) -> dict[str, float]:
    """A member's starting hyperparameters."""
    return {
        "learning_rate": 10 ** rng.uniform(math.log10(3e-4), math.log10(3e-3)),
        "lambda": rng.uniform(0.5, 0.9),
        "pool_fraction": rng.uniform(0.2, 0.8),
        "pfsp": rng.choice([0.0, 2.0]),
    }


def perturb(settings: dict[str, float], rng: random.Random) -> dict[str, float]:
    """PBT's explore step: every continuous setting x0.8 or x1.25 (kept in range), the PFSP switch flipped 1 in 5."""
    out = dict(settings)
    for key in ("learning_rate", "lambda", "pool_fraction"):
        out[key] = settings[key] * rng.choice([0.8, 1.25])
    out["lambda"] = min(max(out["lambda"], 0.0), 1.0)
    out["pool_fraction"] = min(max(out["pool_fraction"], 0.05), 0.95)
    if rng.random() < 0.2:
        out["pfsp"] = 2.0 if settings["pfsp"] == 0.0 else 0.0
    return out


def round_robin(snapshots: list[str], openings: list, depth: int, seed_base: int) -> list[float]:
    """Each member's points per game against all the others: every pair plays each opening from both seats."""
    players = []
    for snapshot in snapshots:
        network = WeightVector.from_json(snapshot)
        players.append(evaluator(network.weights, network.layer_sizes, depth))
    points = [0.0] * len(players)
    games = [0] * len(players)
    for k, (a, b) in enumerate(itertools.combinations(range(len(players)), 2)):
        played = play_pairing(players[a], players[b], openings, seed_base + 1000 * k)
        scored = sum(pair_points(played))
        points[a] += scored
        points[b] += len(played) - scored
        games[a] += len(played)
        games[b] += len(played)
    return [p / g for p, g in zip(points, games, strict=True)]


def main(
    members: int = 8,
    games: int = 200_000,
    interval: int = 10_000,
    depth: int = 3,
    openings_per_pair: int = 2,
    exploit: bool = True,
    rng_seed: int = 0,
    tags: dict[str, Any] | None = None,
) -> str:
    rng = random.Random(rng_seed)
    settings = [sample_settings(rng) for _ in range(members)]
    trainers = []
    for i, s in enumerate(settings):
        trainers.append(
            rl.CheckersSelfPlay(
                rng_seed * 1000 + i,
                params={**BASE, **s},
                max_moves_without_capture=MAX_MOVES_WITHOUT_CAPTURE,
                max_plies=MAX_MOVES,
            )
        )
    generations = max(1, games // interval)
    layer_sizes = WeightVector.from_json(trainers[0].snapshot()).layer_sizes
    config = {
        "representation": "td_lambda",
        "game": "checkers",
        "interface": INTERFACE,
        "search_depth": depth,
        "paradigm": "reinforcement_learning",
        "layer_sizes": list(layer_sizes),
        "params": BASE,
        "population": members,
        "games_per_member": games,
        "interval": interval,
        "exploit": exploit,
        "training": f"population-based self-play ({members} members)"
        if exploit
        else f"random search ({members} members)",
        "initial_settings": settings,
        "max_moves": MAX_MOVES,
        "opponents": list(OPPONENTS),
        "rng_seed": rng_seed,
        **(tags or {}),
    }

    class Counters:
        episodes = 0
        steps = 0

    counters = Counters()
    cost = TrainingCostMeter(population_size=members, fitness=counters)
    all_openings = list(ballot())
    log: list[dict[str, Any]] = []
    with recorded_run(config) as run:
        control = run.control_callback(cost)
        for generation in range(generations):
            for trainer in trainers:
                # in chunks, checking for a pause between them: a generation is members x interval games, far too
                # long to wait for when someone needs the machine back
                for chunk in range(0, interval, PAUSE_CHECK_GAMES):
                    stats = trainer.train(min(PAUSE_CHECK_GAMES, interval - chunk))
                    counters.steps += round(stats["mean_plies"] * stats["games"])
                    counters.episodes += stats["games"]
                    control(None)
            snapshots = [t.snapshot() for t in trainers]
            openings = rng.sample(all_openings, openings_per_pair)
            scores = round_robin(snapshots, openings, depth, SEED_BASE + generation * 100_000)
            ranked = sorted(range(members), key=lambda i: -scores[i])
            best = ranked[0]
            replaced = []
            if exploit and generation < generations - 1:
                quarter = max(1, members // 4)
                for loser in ranked[-quarter:]:
                    winner = rng.choice(ranked[:quarter])
                    trainers[loser].set_weights(list(WeightVector.from_json(snapshots[winner]).weights))
                    settings[loser] = perturb(settings[winner], rng)
                    for key, value in settings[loser].items():
                        trainers[loser].set_param(key, value)
                    replaced.append((loser, winner))
            # The whole population's story, for the Learn chapter's figure: scores, settings, who copied whom.
            log.append(
                {
                    "generation": generation,
                    "scores": scores,
                    "settings": [dict(s) for s in settings],
                    "copied": replaced,
                }
            )
            run.artifacts.put_program(f"{run.run_id}-population", json.dumps(log).encode("utf-8"))
            champion_ref = f"{run.run_id}-gen{generation}"
            run.artifacts.put_program(champion_ref, snapshots[best].encode("utf-8"))
            held_out = None
            if generation % 5 == 0 or generation == generations - 1:
                held_out = monitor_score(
                    WeightVector.from_json(snapshots[best]), OPPONENTS, games=MONITOR_GAMES, depth=depth
                )
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=generation,
                    timestamp=time.time(),
                    best_fitness=scores[best],
                    mean_fitness=sum(scores) / members,
                    worst_fitness=scores[ranked[-1]],
                    diversity=_spread(settings),
                    champion_ref=champion_ref,
                    held_out_score=held_out,
                    extras={
                        "games": float(counters.episodes),
                        "env_steps": float(counters.steps),
                        "replaced": float(len(replaced)),
                        "best_member": float(best),
                        **{f"best_{k}": v for k, v in settings[best].items()},
                    },
                )
            )
            cost.on_generation(None)
            control(None)
            print(
                f"gen {generation:>2}  games/member {(generation + 1) * interval:>7,}  best #{best} {scores[best]:.3f}  "
                f"mean {sum(scores) / members:.3f}  {json.dumps({k: round(v, 4) for k, v in settings[best].items()})}"
                + (f"  held-out {held_out:+.3f}" if held_out is not None else ""),
                flush=True,
            )
        run.set_training_summary(cost)
    return run.run_id


def _spread(settings: list[dict[str, float]]) -> float:
    """How varied the population's learning rates still are (std of log10): PBT narrows it as winners are copied."""
    logs = [math.log10(s["learning_rate"]) for s in settings]
    mean = sum(logs) / len(logs)
    return math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population-based training for Checkers self-play.")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--games", type=int, default=200_000, help="self-play games per member")
    parser.add_argument("--interval", type=int, default=10_000, help="games per member between round robins")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--openings", type=int, default=2, help="ballot openings per pair in each round robin")
    parser.add_argument("--no-exploit", action="store_true", help="the control: same population, nothing copied")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()
    main(
        members=args.members,
        games=args.games,
        interval=args.interval,
        depth=args.depth,
        openings_per_pair=args.openings,
        exploit=not args.no_exploit,
        rng_seed=args.rng_seed,
        tags={"experiment": args.experiment} if args.experiment else None,
    )
