"""Fitness evaluation for LinearProgram genomes.

Fitness is always "higher is better" throughout this package -- for the regression evaluator
below that means -MSE (0 is a perfect fit, more negative is worse) -- so selection, telemetry, and
anything downstream can treat max(fitness) as best without needing to know an error metric is
underneath. See docs/design/0001 (FitnessEvaluator) and docs/design/0003 (dataset vs. simulation
evaluators, sharing this same interface).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from evolve.genome import LinearProgram


class FitnessEvaluator(Protocol):
    def evaluate(self, program: LinearProgram) -> float: ...


class SymbolicRegressionFitness:
    """Dataset-based FitnessEvaluator: -MSE of program.output(x) against a target function."""

    def __init__(
        self, target: Callable[[float], float], inputs: Sequence[float], output_register: int = 0
    ):
        self._output_register = output_register
        self._samples = [(x, target(x)) for x in inputs]

    def evaluate(self, program: LinearProgram) -> float:
        squared_errors = [
            (program.output([x], self._output_register) - y) ** 2 for x, y in self._samples
        ]
        return -sum(squared_errors) / len(squared_errors)
