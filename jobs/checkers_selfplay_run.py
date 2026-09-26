"""Checkers by self-play, recorded to telemetry (docs/design/0010 Phase 4).

A position-value network learns by TD(λ) from games against itself (`rl.CheckersSelfPlay`, the Rust loop in
libs/rl/rust/envs/src/selfplay.rs) -- no opponents, no fitness function, no population. It is the same kind of
network the evolved Checkers runs produce (a `WeightVector`, 32 -> H -> 1, tanh; `evolve`'s wire format), used the
same way: scored from the side to move and searched `search_depth` plies. So its champions join the versus
leaderboard (jobs/evaluate_versus.py) and the Checkers page exactly as the evolved ones do.

Each iteration trains `games_per_iteration` games and records, in the fields every run has (Decision 3):
- best/mean/worst fitness: the *return* of self-play -- the first mover's mean outcome (+1 win, 0 draw, -1 loss)
  over the iteration's games (it hovers near 0 by symmetry; a drift shows a first-move advantage being learned);
- held-out score: every `held_out_every` iterations, the network searching `search_depth` plies against the fixed
  opponents the evolved runs are monitored with (jobs/checkers_training.py's monitor_score: win 1 / draw 0 / loss -1,
  opponent seeds training never uses) -- the curve to watch;
- extras: TD loss, mean game length, draws, epsilon, pool games, and the network's value of the start position
  and of a position a king up.

  uv run python jobs/checkers_selfplay_run.py [--iterations 100] [--games-per-iteration 1000] [--depth 3]
                                             [--param NAME=VALUE ...] [--rng-seed 0]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import rl
from checkers_training import MAX_MOVES, MONITOR_GAMES, monitor_score
from costs import TrainingCostMeter
from evolve import WeightVector
from rl_run import parse_params
from run_context import recorded_run
from telemetry import GenerationStats

INTERFACE = "checkers/board32.v1+evaluate1ply.v1"
OPPONENTS = ("random", "material-1", "material-2")  # the evolved runs' monitor set (checkers_neuro_run.py)
MAX_MOVES_WITHOUT_CAPTURE = 40  # games.checkers.Checkers' default: the draw rule every Checkers game here uses


class _Counters:
    """What TrainingCostMeter reads from a fitness evaluator (`episodes`, `steps`): games and plies here."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0


def main(
    iterations: int = 100,
    games_per_iteration: int = 1000,
    depth: int = 3,
    params: dict[str, float] | None = None,
    rng_seed: int = 0,
    held_out_every: int = 10,
    tags: dict[str, Any] | None = None,
) -> str:
    """Trains one run, records it to telemetry, and returns its run_id."""
    params = params or {}
    trainer = rl.CheckersSelfPlay(
        rng_seed, params=params, max_moves_without_capture=MAX_MOVES_WITHOUT_CAPTURE, max_plies=MAX_MOVES
    )
    layer_sizes = WeightVector.from_json(trainer.snapshot()).layer_sizes
    config = {
        "representation": "td_lambda",
        "game": "checkers",
        # a position evaluator, searched `search_depth` plies -- the interface the evolved evaluators use
        "interface": INTERFACE,
        "search_depth": depth,
        "paradigm": "reinforcement_learning",
        "layer_sizes": list(layer_sizes),
        "params": params,
        "iterations": iterations,
        "games_per_iteration": games_per_iteration,
        "max_moves": MAX_MOVES,
        "opponents": list(OPPONENTS),  # monitored against, never trained against
        "training": "self-play" + (", opponent pool" if params.get("pool_every") else ""),
        "held_out_every": held_out_every,
        "rng_seed": rng_seed,
        **(tags or {}),
    }
    counters = _Counters()
    cost = TrainingCostMeter(population_size=1, fitness=counters)

    with recorded_run(config) as run:
        control = run.control_callback(cost)
        for iteration in range(iterations):
            stats = trainer.train(games_per_iteration)
            counters.episodes = stats["total_games"]
            counters.steps += round(stats["mean_plies"] * stats["games"])
            first_mover_return = (stats["first_wins"] - stats["second_wins"]) / stats["games"]

            champion_ref = f"{run.run_id}-gen{iteration}"
            snapshot = trainer.snapshot()
            run.artifacts.put_program(champion_ref, snapshot.encode("utf-8"))
            held_out = None
            if iteration % held_out_every == 0 or iteration == iterations - 1:
                held_out = monitor_score(WeightVector.from_json(snapshot), OPPONENTS, games=MONITOR_GAMES, depth=depth)
            start_value, king_up_value = trainer.probe()
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=iteration,
                    timestamp=time.time(),
                    best_fitness=first_mover_return,
                    mean_fitness=first_mover_return,
                    worst_fitness=first_mover_return,
                    diversity=0.0,  # a greedy value player has no policy entropy to report
                    champion_ref=champion_ref,
                    held_out_score=held_out,
                    extras={
                        "env_steps": float(counters.steps),
                        "games": float(stats["total_games"]),
                        "td_loss": stats["loss"],
                        "mean_game_plies": stats["mean_plies"],
                        "draw_rate": stats["draws"] / stats["games"],
                        "epsilon": stats["epsilon"],
                        "pool_games": float(stats["pool_games"]),
                        "value_start": start_value,
                        "value_king_up": king_up_value,
                    },
                )
            )
            cost.on_generation(None)
            control(None)
            if held_out is not None:
                print(
                    f"iter {iteration:>3}  games {stats['total_games']:>7,}  loss {stats['loss']:.4f}  "
                    f"plies {stats['mean_plies']:.0f}  held-out {held_out:+.3f}",
                    flush=True,
                )
        history = run.set_training_summary(cost)

    last = history[-1]
    print(f"status=completed  {len(history)} iterations, {counters.episodes:,} games; held-out {last.held_out_score}")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkers by TD(lambda) self-play, recorded to telemetry.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=3, help="search depth for monitoring and the leaderboard")
    parser.add_argument("--held-out-every", type=int, default=10)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="a self-play parameter")
    parser.add_argument("--experiment", default=None, help="tag recorded in the run config (kept off the leaderboard)")
    args = parser.parse_args()
    main(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        depth=args.depth,
        params=parse_params(args.param),
        rng_seed=args.rng_seed,
        held_out_every=args.held_out_every,
        tags={"experiment": args.experiment} if args.experiment else None,
    )
