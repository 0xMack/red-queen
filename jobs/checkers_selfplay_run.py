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
                                             [--param NAME=VALUE ...] [--rng-seed 0] [--init-run RUN_ID]

`--param search_depth=N` (N > 1) trains by TD-Leaf(λ): self-play searches N plies and each position learns at its
principal variation's leaf (libs/rl/rust/envs/src/selfplay.rs).

`--algorithm alphazero` trains AlphaZero-style instead (`rl.CheckersAlphaZero`, libs/rl/rust/envs/src/alphazero.rs,
docs/design/0017): a policy-and-value network, PUCT search on every move, the search's visit counts and the game's
outcome as the targets. Its champion is still the trunk and value unit as a `WeightVector` -- an evaluator, recorded and
ranked exactly as above -- and the final iteration also stores the whole two-headed network (`<champion_ref>-net`).
Extras add the value and policy losses, how far search moved the root's value (`search_shift`), the policy's entropy
at the start, and, at held-out iterations, the network playing *by search* against material-2 (`mcts_points`).
`--init-run` then starts the trunk and value unit from a TD run's network, with a uniform policy.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import rl
from arena.costs import TrainingCostMeter
from checkers_training import MAX_MOVES, MONITOR_GAMES, MONITOR_SEED_BASE, monitor_score
from evolve import WeightVector
from rl_run import parse_params
from run_context import recorded_run
from telemetry import GenerationStats, open_stores

INTERFACE = "checkers/board32.v1+evaluate1ply.v1"
ALGORITHMS = {"td_lambda": rl.CheckersSelfPlay, "alphazero": rl.CheckersAlphaZero}
MCTS_OPPONENT, MCTS_GAMES = "material-2", 20  # AlphaZero's own player (search + policy), monitored at held-out points
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
    init_run: str | None = None,
    algorithm: str = "td_lambda",
) -> str:
    """Trains one run, records it to telemetry, and returns its run_id. `init_run` continues from that run's final
    network (same layer sizes) instead of a fresh one -- e.g. TD-Leaf fine-tuning a plain TD champion. `algorithm` is
    `td_lambda` (TD(λ) / TD-Leaf) or `alphazero`."""
    params = params or {}
    alphazero = algorithm == "alphazero"
    trainer = ALGORITHMS[algorithm](
        rng_seed, params=params, max_moves_without_capture=MAX_MOVES_WITHOUT_CAPTURE, max_plies=MAX_MOVES
    )
    if init_run:
        stores = open_stores()
        metrics, artifacts = stores.metrics, stores.artifacts
        parent = WeightVector.from_json(artifacts.get_program(metrics.history(init_run)[-1].champion_ref).decode())
        if alphazero:
            trainer.set_value_network(list(parent.weights), list(parent.layer_sizes))
        else:
            trainer.set_weights(list(parent.weights))
    layer_sizes = WeightVector.from_json(trainer.snapshot()).layer_sizes
    leaf = int(params.get("search_depth", 1))
    simulations = int(params.get("simulations", 50))
    if alphazero:
        training = f"AlphaZero-style self-play, {simulations} search simulations per move"
    else:
        training = (
            "self-play"
            + (", opponent pool" if params.get("pool_every") else "")
            + (", prioritized opponents" if params.get("pfsp") else "")
            + (f", TD-Leaf searching {leaf} plies" if leaf > 1 else "")
        )
    config = {
        "representation": algorithm,
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
        "training": training,
        "held_out_every": held_out_every,
        "rng_seed": rng_seed,
        **(tags or {}),
    }
    if init_run:
        # The run's own cost is only the continuation; the note says where the rest was spent.
        config["init_run"] = init_run
        config["note"] = f"continued from run {init_run[:8]}"

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
            last = iteration == iterations - 1
            if alphazero and last:
                run.artifacts.put_program(f"{champion_ref}-net", trainer.network_json().encode("utf-8"))
            held_out = None
            extras: dict[str, float] = {}
            if iteration % held_out_every == 0 or last:
                held_out = monitor_score(WeightVector.from_json(snapshot), OPPONENTS, games=MONITOR_GAMES, depth=depth)
                if alphazero:
                    extras["mcts_points"] = trainer.points_against(
                        MCTS_OPPONENT, simulations, MCTS_GAMES, MONITOR_SEED_BASE + iteration
                    )
            if alphazero:
                start_value, king_up_value, start_entropy = trainer.probe()
                extras |= {
                    "value_loss": stats["value_loss"],
                    "policy_loss": stats["policy_loss"],
                    # how far the search's value of a position moved from the network's own: what search adds
                    "search_shift": stats["search_shift"],
                    "policy_entropy_start": start_entropy,
                }
            else:
                start_value, king_up_value = trainer.probe()
                extras |= {
                    "epsilon": stats["epsilon"],
                    "pool_games": float(stats["pool_games"]),
                    # how the network fares against its past selves (1 = beats them all): the pool's difficulty
                    "pool_score": stats["pool_score"],
                }
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
                        "value_start": start_value,
                        "value_king_up": king_up_value,
                        **extras,
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
    parser.add_argument("--init-run", default=None, help="continue from this run's final network (same layer sizes)")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="td_lambda")
    args = parser.parse_args()
    main(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        depth=args.depth,
        params=parse_params(args.param),
        rng_seed=args.rng_seed,
        held_out_every=args.held_out_every,
        tags={"experiment": args.experiment} if args.experiment else None,
        init_run=args.init_run,
        algorithm=args.algorithm,
    )
