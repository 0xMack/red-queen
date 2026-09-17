"""Fitness evaluation for LinearProgram genomes.

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
from typing import Protocol

from evolve.genome import LinearProgram


class FitnessEvaluator(Protocol):
    def evaluate(self, program: LinearProgram) -> list[float]: ...


class SymbolicRegressionFitness:
    """Dataset-based FitnessEvaluator: per-sample -squared-error of program.output(x) against a
    target function."""

    def __init__(
        self, target: Callable[[float], float], inputs: Sequence[float], output_register: int = 0
    ):
        self._output_register = output_register
        self._samples = [(x, target(x)) for x in inputs]

    def evaluate(self, program: LinearProgram) -> list[float]:
        return [
            -((program.output([x], self._output_register) - y) ** 2) for x, y in self._samples
        ]
