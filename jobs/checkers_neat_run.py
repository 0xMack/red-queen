"""NEAT-against-Checkers run, wired to telemetry (docs/design/0008): the structure-evolving counterpart of
jobs/checkers_neuro_run.py -- same game, same opponents, same search depth, same hall of fame, same held-out
monitor and cost meter (jobs/checkers_training.py), so the two are directly comparable. The differences are
exactly the ones NEAT is about:

- the genome is a `NeatGenome`, a graph that grows (add connection / split one into a node), instead of a
  fixed-topology `WeightVector`; it starts as every square wired straight to the one output -- a linear
  evaluator -- and earns hidden nodes only where they pay;
- selection is speciation + fitness sharing inside `evolve_neat`, not lexicase -- fitness is the mean over the
  (opponent, seat) matches;
- telemetry carries `extras` (species count, champion hidden nodes / connections), the curves showing the
  structure grow.

The genome is played through the Rust core's `GraphNet` (`NeatGenome.graph_encoding()`), so a NEAT champion
searches as fast as a layered one. Champions are stored as `NeatGenome.to_json()`.

Run with:
  uv run python jobs/checkers_neat_run.py [--generations 60] [--population 60] [--depth 1]
                                          [--opponents random,material-1,material-2] [--hall 0]
                                          [--seed-material 0.0] [--rng-seed 0]
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from collections.abc import Sequence

from checkers_training import (
    INPUTS,
    MAX_MOVES,
    MONITOR_GAMES,
    OpponentPool,
    make_telemetry_callback,
    material_seed_neat,
)
from costs import TrainingCostMeter
from evolve import InnovationTracker, NeatConfig, evolve_neat, initial_genome
from games.checkers_strategies import STRATEGIES
from run_context import recorded_run

POPULATION_SIZE = 60
GENERATIONS = 60
RNG_SEED = 0
DEFAULT_OPPONENTS = ("random", "material-1", "material-2")
# A linear start with 33 connections never splits under the paper's fixed distance threshold (see NeatConfig),
# so aim for a handful of species and let the threshold adapt -- as the Snake NEAT job does. Weight mutation is
# gentler than NEAT's default: perturbing every weight and replacing 10% outright (the defaults) erases a
# well-placed evaluator -- the seeded material one, or a champion -- faster than it can be improved on.
CHECKERS_NEAT_CONFIG = NeatConfig(
    target_species=5, weight_perturb_sigma=0.04, weight_replace_rate=0.01, initial_weight_scale=0.5
)


def main(
    generations: int = GENERATIONS,
    population_size: int = POPULATION_SIZE,
    opponents: Sequence[str] = DEFAULT_OPPONENTS,
    rng_seed: int = RNG_SEED,
    held_out_every: int = 5,
    depth: int = 1,
    hall: int = 0,
    seed_material: float = 0.0,
    margin: bool = False,
    resample: bool = False,
    games_per_opponent: int = 1,
    config: NeatConfig | None = None,
) -> str:
    """Trains one NEAT run, records it to telemetry, returns its run_id."""
    config = config or CHECKERS_NEAT_CONFIG
    run_config = {
        "representation": "neat",
        "game": "checkers",
        "interface": "checkers/board32.v1+evaluate1ply.v1",
        "search_depth": depth,
        "num_inputs": INPUTS,
        "num_outputs": 1,
        "population_size": population_size,
        "generations": generations,
        "max_moves": MAX_MOVES,
        "opponents": list(opponents),
        "hall_of_fame": hall,
        "seeded_fraction": seed_material,
        "selection": "speciation" if config.speciation else "single species (no speciation)",
        "variation": "neat(add_node, add_connection, weight mutation, innovation-aligned crossover)",
        "neat": dataclasses.asdict(config),
        "fitness": "match outcomes + material margin on draws, both seats per opponent"
        if margin
        else "match outcomes, both seats per opponent",
        "resampled_opponents": resample,
        "games_per_opponent": games_per_opponent,
        "held_out_every": held_out_every,
        "rng_seed": rng_seed,
    }

    rng = random.Random(rng_seed)
    tracker = InnovationTracker(first_hidden_id=INPUTS + 2)
    seeded = round(population_size * seed_material)
    population = [
        material_seed_neat(tracker, rng, config.initial_weight_scale)
        if i < seeded
        else initial_genome(INPUTS, 1, tracker, rng, config.initial_weight_scale)
        for i in range(population_size)
    ]
    pool = OpponentPool(
        opponents, depth, hall_size=hall, margin=margin, resample=resample, games_per_opponent=games_per_opponent
    )
    cost = TrainingCostMeter(population_size=population_size, fitness=pool)

    with recorded_run(run_config) as run:
        evolve_neat(
            population,
            tracker,
            pool,
            config,
            generations,
            on_generation=[
                make_telemetry_callback(
                    run.metrics,
                    run.artifacts,
                    run.run_id,
                    opponents,
                    depth,
                    held_out_every,
                    generations - 1,
                    MONITOR_GAMES,
                ),
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
    parser = argparse.ArgumentParser(description="NEAT vs. Checkers opponents, recorded to telemetry.")
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--depth", type=int, default=1, help="alpha-beta plies the evaluator searches (1 = one ply)")
    parser.add_argument(
        "--opponents", default=",".join(DEFAULT_OPPONENTS), help=f"comma list from {sorted(STRATEGIES)}"
    )
    parser.add_argument("--hall", type=int, default=0, help="hall-of-fame size: past champions as extra opponents")
    parser.add_argument(
        "--seed-material", type=float, default=0.0, help="fraction of the population started as a material evaluator"
    )
    parser.add_argument("--margin", action="store_true", help="score draws by the material edge held")
    parser.add_argument(
        "--resample", action="store_true", help="fresh opponent games every generation (nothing to memorize)"
    )
    parser.add_argument(
        "--games-per-opponent", type=int, default=1, help="games each opponent plays per seat per generation"
    )
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--held-out-every", type=int, default=5)
    args = parser.parse_args()
    main(
        args.generations,
        args.population,
        tuple(args.opponents.split(",")),
        args.rng_seed,
        args.held_out_every,
        args.depth,
        args.hall,
        args.seed_material,
        args.margin,
        args.resample,
        args.games_per_opponent,
    )
