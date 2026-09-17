"""Selection strategies: pick a parent from an evaluated population.

Genome-generic (docs/design/0003) -- operates only on fitness values, never on genome internals,
so the same strategy works for linear GP, tree GP, or an ES weight-vector population.
"""

from __future__ import annotations

import random
from typing import Protocol, TypeVar

Genome = TypeVar("Genome")


class SelectionStrategy(Protocol[Genome]):
    def select(self, population: list[Genome], fitnesses: list[float], rng: random.Random) -> Genome: ...


class TournamentSelection:
    """Selects the fittest of `k` uniformly-random individuals each call."""

    def __init__(self, k: int = 3):
        self._k = k

    def select(self, population: list[Genome], fitnesses: list[float], rng: random.Random) -> Genome:
        indices = [rng.randrange(len(population)) for _ in range(self._k)]
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]
