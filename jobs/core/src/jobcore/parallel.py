"""Score a generation's genomes across processes (`evolve.fitness.evaluate_all` looks for `evaluate_many`).

Every fitness evaluator here is deterministic given its state, so a genome's fitness is the same in a worker as in
the main process -- a parallel run reproduces a serial one exactly, it just uses the cores. The evaluator is pickled
to the workers each generation, so state that changes between generations (a hall of fame, a resampled game set)
must live on the evaluator, and its `on_generation` still runs in the main process.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def _evaluate_chunk(args: tuple[Any, list[Any]]) -> list[list[float]]:
    evaluator, genomes = args
    return [evaluator.evaluate(g) for g in genomes]


class ProcessPoolEvaluator:
    def __init__(self, inner: Any, workers: int | None = None):
        self.inner = inner
        self.workers = workers or os.cpu_count() or 1
        self._executor = ProcessPoolExecutor(max_workers=self.workers)

    def evaluate(self, genome: Any) -> list[float]:
        return self.inner.evaluate(genome)

    def evaluate_many(self, population: list[Any]) -> list[list[float]]:
        # Many small chunks, not one per worker: matches vary a lot in length, so this balances the load.
        size = max(1, len(population) // (self.workers * 3))
        chunks = [(self.inner, population[i : i + size]) for i in range(0, len(population), size)]
        return [row for rows in self._executor.map(_evaluate_chunk, chunks) for row in rows]

    def __getattr__(self, name: str) -> Any:  # episodes/steps counters etc. come from the wrapped evaluator
        return getattr(self.inner, name)
