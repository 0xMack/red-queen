"""Selection strategies: pick a parent from an evaluated population.

Genome-generic (docs/design/0003) -- operates only on fitness values, never on genome internals,
so the same strategy works for linear GP, tree GP, or an ES weight-vector population. Each
individual's fitness is a **list of per-test-case values**, not a single scalar (see fitness.py) --
strategies that only care about overall quality (TournamentSelection) reduce it themselves;
LexicaseSelection uses the per-case breakdown directly, which is the entire point of it.
"""

from __future__ import annotations

import random
import statistics
from typing import Protocol, TypeVar

Genome = TypeVar("Genome")


class SelectionStrategy(Protocol[Genome]):
    def select(
        self, population: list[Genome], case_fitnesses: list[list[float]], rng: random.Random
    ) -> Genome: ...


class TournamentSelection:
    """Selects the fittest (by mean fitness across cases) of `k` uniformly-random individuals."""

    def __init__(self, k: int = 3):
        self._k = k

    def select(
        self, population: list[Genome], case_fitnesses: list[list[float]], rng: random.Random
    ) -> Genome:
        indices = [rng.randrange(len(population)) for _ in range(self._k)]
        best_idx = max(indices, key=lambda i: statistics.fmean(case_fitnesses[i]))
        return population[best_idx]


class LexicaseSelection:
    """Epsilon lexicase selection (Spector-lab style): filters the candidate pool case-by-case,
    in random order, keeping only individuals within `epsilon` of the best remaining fitness on
    each case, until one candidate remains or every case has been used (then picks randomly among
    what's left).

    Epsilon (not strict equality) matters because fitness here is continuous -- exact ties are
    rare, so strict lexicase would usually just collapse to "whoever's best on one random case."
    The default `epsilon=None` computes the per-case median absolute deviation (MAD) of the
    current candidates' values, the standard choice for continuous-valued problems.

    Unlike aggregate/tournament selection, lexicase can and will prefer a "specialist" (excellent
    on some cases, terrible on others) over a "generalist" with a better mean fitness, whenever
    that specialist's strong case happens to be evaluated first -- a real tradeoff, not a bug: it
    trades a bias toward consistency for a bias toward preserving diverse strategies. See
    docs/design/0003 and the tournament-vs-lexicase comparison in notebooks/.
    """

    def __init__(self, epsilon: float | None = None):
        self._epsilon = epsilon

    def select(
        self, population: list[Genome], case_fitnesses: list[list[float]], rng: random.Random
    ) -> Genome:
        candidates = list(range(len(population)))
        num_cases = len(case_fitnesses[0])
        case_order = list(range(num_cases))
        rng.shuffle(case_order)

        for case in case_order:
            if len(candidates) == 1:
                break
            values = [case_fitnesses[i][case] for i in candidates]
            best = max(values)
            epsilon = self._epsilon if self._epsilon is not None else _median_absolute_deviation(values)
            candidates = [i for i in candidates if case_fitnesses[i][case] >= best - epsilon]

        return population[rng.choice(candidates)]


def _median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median([abs(v - center) for v in values])
