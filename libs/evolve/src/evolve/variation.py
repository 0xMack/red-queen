"""Variation strategies: produce one offspring from selected parents.

Unlike selection, variation is necessarily representation-specific (crossover on an instruction
sequence looks nothing like Gaussian noise on a weight vector) -- this is the interface future
representations (tree GP, ES, LLM-driven mutation) plug into as peers. See docs/design/0003
"Synthesis: what's shared, what stays separate".
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Protocol, TypeVar

from evolve.genome import LinearProgram, random_instruction

Genome = TypeVar("Genome")


class VariationStrategy(Protocol[Genome]):
    def vary(self, parents: list[Genome], rng: random.Random) -> Genome: ...


class LinearCrossoverMutation:
    """Single-point instruction crossover between two parents, then per-instruction mutation."""

    def __init__(self, mutation_rate: float = 0.1):
        self._mutation_rate = mutation_rate

    def vary(self, parents: list[LinearProgram], rng: random.Random) -> LinearProgram:
        a = parents[0]
        b = parents[1] if len(parents) > 1 else parents[0]
        n = len(a.instructions)
        point = rng.randrange(n) if n > 1 else 0
        child_instructions = list(a.instructions[:point] + b.instructions[point:])

        for i in range(len(child_instructions)):
            if rng.random() < self._mutation_rate:
                child_instructions[i] = random_instruction(a.num_registers, a.num_inputs, len(a.ops), rng)

        return replace(a, instructions=tuple(child_instructions))
