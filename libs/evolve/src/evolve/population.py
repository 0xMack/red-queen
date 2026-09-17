"""The core, telemetry-agnostic evolution loop.

Deliberately has zero dependency on libs/telemetry -- see docs/design/0001 "Decoupling from
telemetry (same shape, one layer up)". `on_generation` callbacks receive a GenerationSummary
(this module's own type, holding the real champion genome, not a stored ref) so `evolve()` can be
unit-tested with no filesystem/SQLite at all. A separate adapter (see jobs/) converts a
GenerationSummary into telemetry.GenerationStats for anyone who wants a run actually recorded.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from evolve.fitness import FitnessEvaluator
from evolve.selection import SelectionStrategy
from evolve.variation import VariationStrategy

Genome = TypeVar("Genome")


@dataclass(frozen=True, slots=True)
class GenerationSummary(Generic[Genome]):
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    champion: Genome


GenerationCallback = Callable[[GenerationSummary], None]


def evolve(
    initial_population: list[Genome],
    fitness: FitnessEvaluator,
    selection: SelectionStrategy,
    variation: VariationStrategy,
    generations: int,
    on_generation: Iterable[GenerationCallback] = (),
    elitism: int = 1,
    rng: random.Random | None = None,
) -> list[Genome]:
    """Runs `generations` rounds of evaluate -> report -> select+vary, returning the final population.

    `elitism` copies the top N genomes unchanged into the next generation so a lucky champion
    can't be lost to selection noise.
    """
    rng = rng or random.Random()
    population = list(initial_population)
    callbacks = list(on_generation)

    for generation in range(generations):
        # per-test-case fitness (needed by LexicaseSelection); `aggregate` reduces it to one
        # scalar per individual for elitism ranking and the generation summary/telemetry.
        case_fitnesses = [fitness.evaluate(genome) for genome in population]
        aggregate = [statistics.fmean(cf) for cf in case_fitnesses]
        ranked = sorted(range(len(population)), key=lambda i: aggregate[i], reverse=True)

        summary: GenerationSummary[Genome] = GenerationSummary(
            generation=generation,
            best_fitness=aggregate[ranked[0]],
            mean_fitness=statistics.fmean(aggregate),
            worst_fitness=aggregate[ranked[-1]],
            diversity=statistics.pstdev(aggregate) if len(aggregate) > 1 else 0.0,
            champion=population[ranked[0]],
        )
        for callback in callbacks:
            callback(summary)

        next_population = [population[i] for i in ranked[:elitism]]
        while len(next_population) < len(population):
            parents = [
                selection.select(population, case_fitnesses, rng),
                selection.select(population, case_fitnesses, rng),
            ]
            next_population.append(variation.vary(parents, rng))
        population = next_population

    return population
