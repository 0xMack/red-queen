"""NEAT: a graph whose *structure* evolves (docs/design/0008) -- the structure-evolving counterpart to
neuroevolution, on the same games, interfaces, seed strategies, held-out monitoring and cost meter, so the two are
directly comparable. Selection is speciation + fitness sharing inside `evolve_neat`; per-generation `extras` carry
the species count and the champion's size, the curves that show structure growing. Champions are stored as
`NeatGenome.to_json()`."""

from __future__ import annotations

import dataclasses
import random

from arena.checkers import INTERFACE
from arena.costs import TrainingCostMeter
from arena.seeding import SeedStrategy
from arena.snake import MONITOR_SEEDS
from evolve import InnovationTracker, NeatConfig, SimulationFitnessEvaluator, evolve_neat, initial_genome
from games import interfaces
from jobcore import Sink, recorded_run
from jobcore.algorithms.evolution import CheckersNeatParams, SnakeNeatParams
from jobcore.specs import TrainSpec

from trainer import checkers, snake
from trainer.registry import adapter, held_out_every

# The paper's fixed threshold never splits Snake's 36-gene starting genomes (see NeatConfig), so aim for a
# handful of species and let the threshold adapt.
SNAKE_NEAT_CONFIG = NeatConfig(target_species=6)
# A linear start with 33 connections never splits under the paper's fixed distance threshold either, so the same; and
# weight mutation gentler than NEAT's default: perturbing every weight and replacing 10% outright (the defaults) erases
# a well-placed evaluator -- the seeded material one, or a champion -- faster than it can be improved on.
CHECKERS_NEAT_CONFIG = NeatConfig(
    target_species=5, weight_perturb_sigma=0.04, weight_replace_rate=0.01, initial_weight_scale=0.5
)


def _selection(config: NeatConfig) -> str:
    return "speciation" if config.speciation else "single species (no speciation)"


VARIATION = "neat(add_node, add_connection, weight mutation, innovation-aligned crossover)"


@adapter("neat", "snake")
def train_snake(spec: TrainSpec, params: SnakeNeatParams, sink: Sink) -> str:
    generations, every = spec.budget_amount, held_out_every(spec, 10)
    neat = dataclasses.replace(SNAKE_NEAT_CONFIG, **params.neat)
    interface = interfaces.get(spec.interface or snake.DEFAULT_INTERFACE)
    seeds = SeedStrategy.parse(params.seeds, rng_seed=spec.seed)
    envs = [interface.make_game(seed=seed, **snake.BOARD) for seed in seeds.initial()]
    num_inputs = len(interface.observer.feature_names(envs[0]))
    num_outputs = interface.action.num_outputs

    config = {
        "representation": "neat",
        "game": "snake",
        "interface": interface.id,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "population_size": params.population_size,
        "generations": generations,
        "max_steps": params.max_steps,
        "selection": _selection(neat),
        "variation": VARIATION,
        "neat": dataclasses.asdict(neat),
        "benchmark": "games.snake (10x10)",
        **seeds.config(),
        "held_out_every": every,
        "monitor_seeds": [MONITOR_SEEDS[0], MONITOR_SEEDS[-1]],
        "rng_seed": spec.seed,
        **spec.tags,
    }

    rng = random.Random(spec.seed)
    tracker = InnovationTracker(first_hidden_id=num_inputs + 1 + num_outputs)
    population = [
        initial_genome(num_inputs, num_outputs, tracker, rng, neat.initial_weight_scale)
        for _ in range(params.population_size)
    ]
    fitness = SimulationFitnessEvaluator(
        envs=envs, act=snake.make_act(interface), max_steps=params.max_steps, rollout=snake.make_rollout(interface)
    )
    cost = TrainingCostMeter(population_size=params.population_size, fitness=fitness)

    with recorded_run(config, sink) as run:
        evolve_neat(
            population,
            tracker,
            fitness,
            neat,
            generations,
            on_generation=[
                snake.make_telemetry_callback(
                    run.metrics, run.artifacts, run.run_id, held_out=(interface, every, generations - 1)
                ),
                *([snake.make_resample_callback(fitness, seeds, interface)] if seeds.resamples else []),
                cost.on_generation,
                run.control_callback(cost),
            ],
            rng=rng,
        )
        run.set_training_summary(cost)
    return run.run_id


@adapter("neat", "checkers")
def train_checkers(spec: TrainSpec, params: CheckersNeatParams, sink: Sink) -> str:
    generations, every = spec.budget_amount, held_out_every(spec, 5)
    neat = dataclasses.replace(CHECKERS_NEAT_CONFIG, **params.neat)
    config = {
        "representation": "neat",
        "game": "checkers",
        "interface": INTERFACE,
        "search_depth": params.depth,
        "num_inputs": checkers.INPUTS,
        "num_outputs": 1,
        "population_size": params.population_size,
        "generations": generations,
        "max_moves": checkers.MAX_MOVES,
        "opponents": list(params.opponents),
        "hall_of_fame": params.hall,
        "seeded_fraction": params.seed_material,
        "selection": _selection(neat),
        "variation": VARIATION,
        "neat": dataclasses.asdict(neat),
        "fitness": "match outcomes + material margin on draws, both seats per opponent"
        if params.margin
        else "match outcomes, both seats per opponent",
        "resampled_opponents": params.resample,
        "games_per_opponent": params.games_per_opponent,
        "held_out_every": every,
        "rng_seed": spec.seed,
        **spec.tags,
    }

    rng = random.Random(spec.seed)
    tracker = InnovationTracker(first_hidden_id=checkers.INPUTS + 2)
    seeded = round(params.population_size * params.seed_material)
    population = [
        checkers.material_seed_neat(tracker, rng, neat.initial_weight_scale)
        if i < seeded
        else initial_genome(checkers.INPUTS, 1, tracker, rng, neat.initial_weight_scale)
        for i in range(params.population_size)
    ]
    pool = checkers.OpponentPool(
        params.opponents,
        params.depth,
        hall_size=params.hall,
        margin=params.margin,
        resample=params.resample,
        games_per_opponent=params.games_per_opponent,
    )
    cost = TrainingCostMeter(population_size=params.population_size, fitness=pool)

    with recorded_run(config, sink) as run:
        evolve_neat(
            population,
            tracker,
            pool,
            neat,
            generations,
            on_generation=[
                checkers.make_telemetry_callback(
                    run.metrics,
                    run.artifacts,
                    run.run_id,
                    params.opponents,
                    params.depth,
                    every,
                    generations - 1,
                    params.monitor_games,
                ),
                pool.on_generation,
                cost.on_generation,
                run.control_callback(cost),
            ],
            rng=rng,
        )
        run.set_training_summary(cost)
    return run.run_id
