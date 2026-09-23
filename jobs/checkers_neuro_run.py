"""Neuroevolution-against-Checkers run, wired to telemetry -- the first *multi-agent* training job, and
the real test of the strategy/match framework (docs/design/0006): a genome is not a policy over a fixed
action space but a *position evaluator*, and fitness is match outcomes, not a simulation score.

The genome is a `WeightVector` (32 -> HIDDEN -> 1, tanh): given the board from a player's perspective
(`games.checkers`' 32-square encoding) it outputs how good that position is *for the player to move*. A
player is that evaluator plus a search (`--depth` plies of alpha-beta in the Rust core; 1 = one ply of
lookahead, exactly what the observation encoding was designed for) -- the classic recipe for learning
Checkers, neuroevolution supplying the evaluation and minimax the lookahead. Because the network runs in Rust
too, a whole match is a handful of native calls, not one Python forward pass per candidate move.

Fitness is `evolve.MatchFitnessEvaluator` (env-aware mode) against `jobs/checkers_training.OpponentPool`: fixed
opponents (the static strategies in `games.checkers_strategies`) plus, with `--hall`, a hall of fame of this
run's own past champions. Each genome plays every opponent from both seats, one value per (opponent, seat), so
LexicaseSelection can prefer a genome that beats the hard opponent from one seat over one that only beats the
easy ones. `--seed-material` starts that fraction of the population as a noisy material evaluator.

The champion is also scored every --held-out-every generations on games it never trained on (different
opponent rng seeds, same pool): the mean of win=1 / draw=0 / loss=-1 over MONITOR_GAMES games per opponent,
alternating seats -- the overfitting curve, same idea as snake_neuro_run.py's. The NEAT counterpart is
jobs/checkers_neat_run.py.

Run with:
  uv run python jobs/checkers_neuro_run.py [--generations 60] [--population 40] [--hidden 8] [--depth 1]
                                           [--opponents random,material-1,material-2] [--hall 0]
                                           [--seed-material 0.0] [--rng-seed 0]
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from typing import Any

from checkers_training import (
    INPUTS,
    MAX_MOVES,
    MONITOR_GAMES,
    MONITOR_SEED_BASE,
    OpponentPool,
    make_act,
    make_telemetry_callback,
    material_seed_weights,
    monitor_score,
    opponent_pool,
    strategy_factory,
)
from costs import TrainingCostMeter
from evolve import GaussianMutation, LexicaseSelection, evolve, random_weight_vector
from games.checkers_strategies import STRATEGIES
from parallel import ProcessPoolEvaluator
from run_context import recorded_run

HIDDEN = 8
POPULATION_SIZE = 40
GENERATIONS = 60
RNG_SEED = 0
DEFAULT_OPPONENTS = ("random", "material-1", "material-2")

__all__ = ["INPUTS", "MAX_MOVES", "MONITOR_GAMES", "MONITOR_SEED_BASE", "make_act", "monitor_score", "opponent_pool", "strategy_factory", "main"]


def main(
    generations: int = GENERATIONS,
    population_size: int = POPULATION_SIZE,
    hidden: int = HIDDEN,
    opponents: Sequence[str] = DEFAULT_OPPONENTS,
    rng_seed: int = RNG_SEED,
    held_out_every: int = 5,
    depth: int = 1,
    hall: int = 0,
    seed_material: float = 0.0,
    margin: bool = False,
    resample: bool = False,
    games_per_opponent: int = 1,
    sigma: float = 0.2,
    mutation_rate: float = 1.0,
    tags: dict[str, Any] | None = None,
    workers: int = 1,
    opening_plies: int = 0,
) -> str:
    """Trains one run, records it to telemetry, returns its run_id."""
    layer_sizes = (INPUTS, hidden, 1)
    config = {
        "representation": "neuroevolution",
        "game": "checkers",
        # How the champion is used: a position evaluator (score = forward(observation)[0], from the
        # perspective of the player to move), searched `search_depth` plies -- not a fixed action space.
        "interface": "checkers/board32.v1+evaluate1ply.v1",
        "search_depth": depth,
        "layer_sizes": list(layer_sizes),
        "population_size": population_size,
        "generations": generations,
        "max_moves": MAX_MOVES,
        "opponents": list(opponents),
        "hall_of_fame": hall,
        "seeded_fraction": seed_material,
        "selection": "lexicase",
        "variation": f"gaussian_mutation(sigma={sigma}, rate={mutation_rate})",
        "fitness": "match outcomes + material margin on draws, both seats per opponent" if margin else "match outcomes, both seats per opponent",
        "resampled_opponents": resample,
        "games_per_opponent": games_per_opponent,
        "opening_plies": opening_plies,
        "held_out_every": held_out_every,
        "rng_seed": rng_seed,
        **(tags or {}),  # e.g. `experiment`/`arm`: kept off the leaderboard, aggregated by the experiment
    }

    rng = random.Random(rng_seed)
    seeded = round(population_size * seed_material)
    population = [
        material_seed_weights(layer_sizes, rng) if i < seeded else random_weight_vector(layer_sizes, rng, scale=0.5)
        for i in range(population_size)
    ]
    pool = OpponentPool(opponents, depth, hall_size=hall, margin=margin, resample=resample, games_per_opponent=games_per_opponent, opening_plies=opening_plies)
    fitness = ProcessPoolEvaluator(pool, workers) if workers > 1 else pool
    cost = TrainingCostMeter(population_size=population_size, fitness=fitness)

    with recorded_run(config) as run:
        evolve(
            population,
            fitness=fitness,
            selection=LexicaseSelection(),
            variation=GaussianMutation(sigma=sigma, rate=mutation_rate),
            generations=generations,
            on_generation=[
                make_telemetry_callback(run.metrics, run.artifacts, run.run_id, opponents, depth, held_out_every, generations - 1, MONITOR_GAMES),
                pool.on_generation,
                cost.on_generation,
                run.control_callback(cost),
            ],
            rng=rng,
        )
        history = run.set_training_summary(cost)
    print(f"status=completed  recorded {len(history)} generations", flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuroevolution vs. Checkers opponents, recorded to telemetry.")
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--depth", type=int, default=1, help="alpha-beta plies the evaluator searches (1 = one ply)")
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS), help=f"comma list from {sorted(STRATEGIES)}")
    parser.add_argument("--hall", type=int, default=0, help="hall-of-fame size: past champions as extra opponents")
    parser.add_argument("--seed-material", type=float, default=0.0, help="fraction of the population started as a material evaluator")
    parser.add_argument("--margin", action="store_true", help="score draws by the material edge held")
    parser.add_argument("--resample", action="store_true", help="fresh opponent games every generation (nothing to memorize)")
    parser.add_argument("--games-per-opponent", type=int, default=1, help="games each opponent plays per seat per generation")
    parser.add_argument("--sigma", type=float, default=0.2, help="Gaussian mutation step")
    parser.add_argument("--mutation-rate", type=float, default=1.0, help="chance each weight is perturbed (small = sparse mutation)")
    parser.add_argument("--experiment", default=None, help="tag recorded in the run config (kept off the leaderboard)")
    parser.add_argument("--arm", default=None, help="which arm of the experiment this run is")
    parser.add_argument("--opening-plies", type=int, default=0, help="random moves opening every fitness game (both seats share one)")
    parser.add_argument("--workers", type=int, default=1, help="processes scoring a generation's genomes (same result as 1, faster)")
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--held-out-every", type=int, default=5)
    args = parser.parse_args()
    main(
        args.generations,
        args.population,
        args.hidden,
        tuple(args.opponents.split(",")),
        args.rng_seed,
        args.held_out_every,
        args.depth,
        args.hall,
        args.seed_material,
        args.margin,
        args.resample,
        args.games_per_opponent,
        args.sigma,
        args.mutation_rate,
        {"experiment": args.experiment, "arm": args.arm} if args.experiment else None,
        args.workers,
        args.opening_plies,
    )
