"""Fitness evaluation, generic over genome representation.

Fitness is always "higher is better" throughout this package -- for the regression evaluator
below that means -squared-error per case (0 is a perfect fit, more negative is worse) -- so
selection, telemetry, and anything downstream can treat max(fitness) as best without needing to
know an error metric is underneath. See docs/design/0001 (FitnessEvaluator) and docs/design/0003
(dataset vs. simulation evaluators, sharing this same interface).

`evaluate()` returns one fitness value **per test case**, not a single aggregate score --
LexicaseSelection (docs/design/0003 "GP frontier") needs the per-case breakdown, not just a mean.
Anything that wants a single scalar (elitism ranking, telemetry summaries) reduces this itself,
same as TournamentSelection does internally.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar

Genome = TypeVar("Genome")


class FitnessEvaluator(Protocol[Genome]):
    def evaluate(self, program: Genome) -> list[float]: ...


def _default_linear_run(program: object, x: float) -> float:
    return program.output([x])  # type: ignore[attr-defined]


class SymbolicRegressionFitness(Generic[Genome]):
    """Dataset-based FitnessEvaluator: per-sample -squared-error of `run(program, x)` against a
    target function.

    `run` defaults to `LinearProgram.output` (single input, output register 0) -- the common case
    in this package so far -- but is genuinely generic: pass e.g. `lambda t, x: t.evaluate(x)` for
    a TreeProgram, or any other representation's single-input evaluation. This class never touches
    genome internals itself (docs/design/0003 "what's shared, what stays separate").
    """

    def __init__(
        self,
        target: Callable[[float], float],
        inputs: Sequence[float],
        run: Callable[[Genome, float], float] = _default_linear_run,
    ):
        self._samples = [(x, target(x)) for x in inputs]
        self._run = run

    def evaluate(self, program: Genome) -> list[float]:
        return [-((self._run(program, x) - y) ** 2) for x, y in self._samples]


def evaluate_all(fitness, population):
    """Every genome's per-case fitness, in population order. An evaluator may offer `evaluate_many(population)` to
    score the whole generation at once (e.g. across processes, jobs/parallel.py); otherwise it is one at a time."""
    batch = getattr(fitness, "evaluate_many", None)
    if batch is not None:
        return list(batch(population))
    return [fitness.evaluate(genome) for genome in population]
