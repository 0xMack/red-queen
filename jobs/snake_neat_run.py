"""NEAT-against-Snake run, wired to telemetry (docs/design/0008).

The structure-evolving counterpart to snake_neuro_run.py -- same game, same interface
(docs/design/0007), same training-seed strategies, same held-out monitoring, same cost meter, so the
two are directly comparable. The differences are exactly the ones NEAT is about:

- the genome is a `NeatGenome` (a graph that grows) instead of a fixed `WeightVector`;
- selection is speciation + fitness sharing inside `evolve_neat`, not lexicase/tournament -- fitness
  is the mean over the training games;
- per-generation telemetry carries `extras` (species count, champion hidden nodes/connections), the
  curves that show structure growing.

Champions are stored as `NeatGenome.to_json()`; `jobs/evaluate.py` and apps/frontend load them back
through `evolve.network_from_json`.

Run with:
  uv run python jobs/snake_neat_run.py [interface_id] [--seeds fixed:5|resample:N] [--generations 250]
                                       [--rng-seed 0] [--experiment NAME] [--no-speciation]
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from typing import Any

from costs import TrainingCostMeter
from evaluate import MONITOR_SEEDS
from evolve import (
    InnovationTracker,
    NeatConfig,
    SimulationFitnessEvaluator,
    evolve_neat,
    initial_genome,
    network_from_json,
)
from games import interfaces
from run_context import recorded_run
from seeding import SeedStrategy
from snake_neuro_run import (
    BOARD,
    DEFAULT_INTERFACE,
    GENERATIONS,
    MAX_STEPS,
    POPULATION_SIZE,
    RNG_SEED,
    make_act,
    make_rollout,
    make_resample_callback,
    make_telemetry_callback,
)

# The paper's fixed threshold never splits Snake's 36-gene starting genomes (see NeatConfig), so aim for a
# handful of species and let the threshold adapt.
SNAKE_NEAT_CONFIG = NeatConfig(target_species=6)


def main(
    interface_id: str = DEFAULT_INTERFACE,
    seed_strategy: str = "fixed:5",
    held_out_every: int = 10,
    generations: int = GENERATIONS,
    rng_seed: int = RNG_SEED,
    tags: dict[str, Any] | None = None,
    config: NeatConfig | None = None,
    population_size: int = POPULATION_SIZE,
    max_steps: int = MAX_STEPS,
) -> str:
    """Trains one NEAT run, records it to telemetry, and returns its run_id. `tags` are extra config
    entries (jobs/snake_experiment.py records `experiment` and `arm`)."""
    config = config or SNAKE_NEAT_CONFIG
    interface = interfaces.get(interface_id)
    seeds = SeedStrategy.parse(seed_strategy, rng_seed=rng_seed)
    envs = [interface.make_game(seed=seed, **BOARD) for seed in seeds.initial()]
    num_inputs = len(interface.observer.feature_names(envs[0]))
    num_outputs = interface.action.num_outputs

    run_config = {
        "representation": "neat",
        "game": "snake",
        "interface": interface.id,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "population_size": population_size,
        "generations": generations,
        "max_steps": max_steps,
        "selection": "speciation"
        if config.speciation
        else "single species (no speciation)",
        "variation": "neat(add_node, add_connection, weight mutation, innovation-aligned crossover)",
        "neat": dataclasses.asdict(config),
        "benchmark": "games.snake (10x10)",
        **seeds.config(),
        "held_out_every": held_out_every,
        "monitor_seeds": [MONITOR_SEEDS[0], MONITOR_SEEDS[-1]],
        "rng_seed": rng_seed,
        **(tags or {}),
    }

    rng = random.Random(rng_seed)
    tracker = InnovationTracker(first_hidden_id=num_inputs + 1 + num_outputs)
    population = [
        initial_genome(
            num_inputs, num_outputs, tracker, rng, config.initial_weight_scale
        )
        for _ in range(population_size)
    ]
    fitness = SimulationFitnessEvaluator(
        envs=envs,
        act=make_act(interface),
        max_steps=max_steps,
        rollout=make_rollout(interface),
    )
    cost = TrainingCostMeter(population_size=population_size, fitness=fitness)

    with recorded_run(run_config) as run:
        evolve_neat(
            population,
            tracker,
            fitness,
            config,
            generations,
            on_generation=[
                make_telemetry_callback(
                    run.registry,
                    run.metrics,
                    run.artifacts,
                    run.run_id,
                    held_out=(interface, held_out_every, generations - 1),
                ),
                *(
                    [make_resample_callback(fitness, seeds, interface)]
                    if seeds.resamples
                    else []
                ),
                cost.on_generation,
                run.control_callback(cost),
            ],
            rng=rng,
        )
        history = run.set_training_summary(cost)

    final = network_from_json(
        run.artifacts.get_program(history[-1].champion_ref).decode("utf-8")
    )
    hidden, connections = final.complexity()
    print(
        f"status={run.registry.get_run(run.run_id).status}, recorded {len(history)} generations"
    )
    print(f"gen 0   best_fitness={history[0].best_fitness:.4f}")
    print(
        f"gen {history[-1].generation:<3} best_fitness={history[-1].best_fitness:.4f}  held_out={history[-1].held_out_score}"
    )
    print(
        f"final champion: {hidden} hidden nodes, {connections} enabled connections, species={history[-1].extras['species']:.0f}"
    )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEAT vs. Snake, recorded to telemetry."
    )
    parser.add_argument(
        "interface", nargs="?", default=DEFAULT_INTERFACE, help="games.interfaces id"
    )
    parser.add_argument(
        "--seeds", default="fixed:5", help="fixed:N or resample:N (jobs/seeding.py)"
    )
    parser.add_argument("--held-out-every", type=int, default=10)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="training episode cap (the leaderboard scores 1000-step games)",
    )
    parser.add_argument(
        "--experiment", default=None, help="tag recorded in the run config"
    )
    parser.add_argument(
        "--no-speciation", action="store_true", help="ablation: one big species"
    )
    args = parser.parse_args()
    main(
        args.interface,
        args.seeds,
        args.held_out_every,
        args.generations,
        args.rng_seed,
        {"experiment": args.experiment} if args.experiment else None,
        dataclasses.replace(SNAKE_NEAT_CONFIG, speciation=not args.no_speciation),
        args.population,
        args.max_steps,
    )
