"""Neuroevolution: a fixed-topology network's weights, evolved (docs/design/0003, 0006, 0007).

- **Snake**: a policy network under a named interface (observer + action adapter, recorded so the champion is always
  run under the observation it was trained for), lexicase selection over the training games (each game a case:
  aggregate selection trades away per-game performance for a better average, notebooks/0001), the game core playing
  every training episode, and the champion's score on games it never trained on every `held_out_every` generations --
  the curve that shows overfitting (with `fixed:5` seeds the champion memorizes its five games).
- **Checkers**: a 32 -> H -> 1 position evaluator, searched `depth` plies by the Rust core; fitness is match outcomes
  against fixed opponents (plus an optional hall of fame of past champions), one value per opponent per seat, so
  lexicase can prefer a genome that beats the hard opponent from one seat (see trainer.checkers).
"""

from __future__ import annotations

import random

from arena.checkers import INTERFACE
from arena.costs import TrainingCostMeter
from arena.seeding import SeedStrategy
from arena.snake import MONITOR_SEEDS
from evolve import (
    GaussianMutation,
    LexicaseSelection,
    SimulationFitnessEvaluator,
    TournamentSelection,
    evolve,
    random_weight_vector,
)
from games import interfaces
from jobcore import ProcessPoolEvaluator, Sink, recorded_run
from jobcore.algorithms.evolution import CheckersNeuroParams, SnakeNeuroParams
from jobcore.specs import TrainSpec

from trainer import checkers, snake
from trainer.registry import adapter, held_out_every


@adapter("neuroevolution", "snake")
def train_snake(spec: TrainSpec, params: SnakeNeuroParams, sink: Sink) -> str:
    generations, every = spec.budget_amount, held_out_every(spec, 10)
    interface = interfaces.get(spec.interface or snake.DEFAULT_INTERFACE)
    seeds = SeedStrategy.parse(params.seeds, rng_seed=spec.seed)
    envs = [interface.make_game(seed=seed, **snake.BOARD) for seed in seeds.initial()]
    layer_sizes = (len(interface.observer.feature_names(envs[0])), params.hidden, interface.action.num_outputs)
    tournament = params.selection == "tournament"

    config = {
        "representation": "neuroevolution",
        "game": "snake",  # apps/frontend's watch page reads this to know what to render
        "interface": interface.id,  # how the champion must be observed/decoded (doc 0007)
        "layer_sizes": list(layer_sizes),
        "population_size": params.population_size,
        "generations": generations,
        "max_steps": params.max_steps,
        "selection": f"tournament(k={params.tournament_k})" if tournament else "lexicase",
        "variation": "gaussian_mutation(sigma=0.2)",
        "benchmark": "games.snake (10x10)",
        **seeds.config(),  # seed_strategy + training_seeds (docs/design/0007: overfitting)
        "held_out_every": every,
        "monitor_seeds": [MONITOR_SEEDS[0], MONITOR_SEEDS[-1]],
        "rng_seed": spec.seed,
        **spec.tags,
    }

    rng = random.Random(spec.seed)
    population = [random_weight_vector(layer_sizes, rng, scale=0.5) for _ in range(params.population_size)]
    fitness = SimulationFitnessEvaluator(
        envs=envs, act=snake.make_act(interface), max_steps=params.max_steps, rollout=snake.make_rollout(interface)
    )
    cost = TrainingCostMeter(population_size=params.population_size, fitness=fitness)

    with recorded_run(config, sink) as run:
        evolve(
            population,
            fitness=fitness,
            selection=TournamentSelection(k=params.tournament_k) if tournament else LexicaseSelection(),
            variation=GaussianMutation(sigma=0.2),
            generations=generations,
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


@adapter("neuroevolution", "checkers")
def train_checkers(spec: TrainSpec, params: CheckersNeuroParams, sink: Sink) -> str:
    generations, every = spec.budget_amount, held_out_every(spec, 5)
    layer_sizes = (checkers.INPUTS, params.hidden, 1)
    config = {
        "representation": "neuroevolution",
        "game": "checkers",
        # How the champion is used: a position evaluator (score = forward(observation)[0], from the
        # perspective of the player to move), searched `search_depth` plies -- not a fixed action space.
        "interface": INTERFACE,
        "search_depth": params.depth,
        "layer_sizes": list(layer_sizes),
        "population_size": params.population_size,
        "generations": generations,
        "max_moves": checkers.MAX_MOVES,
        "opponents": list(params.opponents),
        "hall_of_fame": params.hall,
        "seeded_fraction": params.seed_material,
        "selection": "lexicase",
        "variation": f"gaussian_mutation(sigma={params.sigma}, rate={params.mutation_rate})",
        "fitness": "match outcomes + material margin on draws, both seats per opponent"
        if params.margin
        else "match outcomes, both seats per opponent",
        "resampled_opponents": params.resample,
        "games_per_opponent": params.games_per_opponent,
        "opening_plies": params.opening_plies,
        "held_out_every": every,
        "rng_seed": spec.seed,
        **spec.tags,  # e.g. `experiment`/`arm`: kept off the leaderboard, aggregated by the experiment
    }

    rng = random.Random(spec.seed)
    seeded = round(params.population_size * params.seed_material)
    population = [
        checkers.material_seed_weights(layer_sizes, rng)
        if i < seeded
        else random_weight_vector(layer_sizes, rng, scale=0.5)
        for i in range(params.population_size)
    ]
    pool = checkers.OpponentPool(
        params.opponents,
        params.depth,
        hall_size=params.hall,
        margin=params.margin,
        resample=params.resample,
        games_per_opponent=params.games_per_opponent,
        opening_plies=params.opening_plies,
    )
    fitness = ProcessPoolEvaluator(pool, params.workers) if params.workers > 1 else pool
    cost = TrainingCostMeter(population_size=params.population_size, fitness=fitness)

    with recorded_run(config, sink) as run:
        evolve(
            population,
            fitness=fitness,
            selection=LexicaseSelection(),
            variation=GaussianMutation(sigma=params.sigma, rate=params.mutation_rate),
            generations=generations,
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
