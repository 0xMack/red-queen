"""Evolving a bandit strategy's settings (docs/design/0011, "Evolved strategy"): the one evolution-vs-RL comparison that
fits on a page. A bandit strategy learns *within* a game; evolution tunes *how* it learns, across many games.

The genome is four numbers (an `evolve.WeightVector` of shape (4,)) decoded into ε-greedy's settings: how often it
explores (ε), over how many pulls exploring fades out (decay), how much each payout moves an estimate (the step, alpha),
and how optimistic untried machines start (the initial estimate). Fitness is skill (0 = random, 100 = the best machine
every pull) on fresh *training* games every generation -- seeds 1-9,999, never the leaderboard's held-out 10,000+ --
and every few generations the champion's skill on a fixed monitor set (`arena.bandit.MONITOR_SEEDS`) is recorded as the
run's held-out score, so the run page shows whether evolution is fitting the training games or the game.

Recorded like every evolution run (`representation: evolved_bandit`, `game: bandit`); `jobs/evaluate_bandit.py` then
ranks every completed bandit run's final champion on the held-out games with the hand-tuned strategies.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from typing import Any

from arena.bandit import MONITOR_SEEDS
from arena.costs import TrainingCostMeter
from evolve import GaussianMutation, GenerationSummary, TournamentSelection, WeightVector, evolve
from jobcore import Sink, recorded_run
from jobcore.algorithms.evolution import BanditEvolveParams
from jobcore.specs import TrainSpec
from rl import _native
from telemetry import GenerationStats

from trainer.registry import adapter, held_out_every

INTERFACE = "bandit/none.v1+arm.v1"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def decode(genome: WeightVector) -> dict[str, float]:
    """The four numbers as ε-greedy's settings, each squashed into a sensible range (rounded: they're shown to people)."""
    g = genome.weights
    return {
        "epsilon": round(_sigmoid(g[0]), 3),
        "decay": round(20 + 380 * _sigmoid(g[1])),
        "alpha": round(0.5 * _sigmoid(g[2]), 3),
        "initial": round(2 * _sigmoid(g[3]), 3),
    }


class BanditFitness:
    """Per-game skill (times 100) of the decoded strategy on this generation's training games, every scenario in turn."""

    def __init__(self, scenarios: list[str], rng: random.Random, games_per_scenario: int):
        self.scenarios = scenarios
        self._rng = rng
        self._games = games_per_scenario
        self.episodes = 0
        self.steps = 0
        self.resample()

    def resample(self) -> None:
        self.seeds = [self._rng.randrange(1, 10_000) for _ in range(self._games)]

    def evaluate(self, genome: WeightVector) -> list[float]:
        params = decode(genome)
        cases: list[float] = []
        for scenario in self.scenarios:
            results = _native.bandit_evaluate("epsilon_greedy", scenario, "none.v1", self.seeds, params)
            cases += [r["skill"] * 100 for r in results]
            self.episodes += len(results)
            self.steps += len(results) * (200 if scenario == "drifting" else 100)
        return cases


def monitor_skill(genome: WeightVector, scenarios: list[str]) -> float:
    params = decode(genome)
    skills = [
        r["skill"] * 100
        for scenario in scenarios
        for r in _native.bandit_evaluate("epsilon_greedy", scenario, "none.v1", list(MONITOR_SEEDS), params)
    ]
    return statistics.fmean(skills)


@adapter("bandit_evolve", "bandit")
def train_bandit(spec: TrainSpec, params: BanditEvolveParams, sink: Sink) -> str:
    generations, every, scenarios = spec.budget_amount, held_out_every(spec, 5), list(params.scenarios)
    rng = random.Random(spec.seed)
    fitness = BanditFitness(scenarios, rng, params.games_per_scenario)
    population = [
        WeightVector(tuple(rng.gauss(0.0, 1.5) for _ in range(4)), (4,)) for _ in range(params.population_size)
    ]
    config: dict[str, Any] = {
        "representation": "evolved_bandit",
        "game": "bandit",
        "interface": INTERFACE,
        "strategy": "epsilon_greedy",
        "genome": "epsilon, decay, alpha, initial (4 numbers, squashed)",
        "scenarios": scenarios,
        "population_size": params.population_size,
        "generations": generations,
        "selection": f"tournament(k={params.tournament_k})",
        "variation": f"gaussian_mutation(sigma={params.sigma})",
        "seed_strategy": f"resample:{params.games_per_scenario}",
        "held_out_every": every,
        "monitor_seeds": [MONITOR_SEEDS[0], MONITOR_SEEDS[-1]],
        "rng_seed": spec.seed,
        **spec.tags,
    }
    cost = TrainingCostMeter(population_size=params.population_size, fitness=fitness)

    with recorded_run(config, sink) as run:

        def record(summary: GenerationSummary) -> None:
            champion_ref = f"{run.run_id}-gen{summary.generation}"
            artifact = {
                "genome": list(summary.champion.weights),
                "strategy": "epsilon_greedy",
                "params": decode(summary.champion),
            }
            run.artifacts.put_program(champion_ref, json.dumps(artifact).encode("utf-8"))
            last = summary.generation == generations - 1
            held_out = monitor_skill(summary.champion, scenarios) if summary.generation % every == 0 or last else None
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=summary.generation,
                    timestamp=time.time(),
                    best_fitness=summary.best_fitness,
                    mean_fitness=summary.mean_fitness,
                    worst_fitness=summary.worst_fitness,
                    diversity=summary.diversity,
                    champion_ref=champion_ref,
                    held_out_score=held_out,
                    extras={k: float(v) for k, v in decode(summary.champion).items()},
                )
            )

        evolve(
            population,
            fitness=fitness,
            selection=TournamentSelection(k=params.tournament_k),
            variation=GaussianMutation(sigma=params.sigma),
            generations=generations,
            on_generation=[record, lambda _s: fitness.resample(), cost.on_generation, run.control_callback(cost)],
            rng=rng,
        )
        run.set_training_summary(cost)
    return run.run_id
