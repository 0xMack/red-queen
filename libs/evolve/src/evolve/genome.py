"""A pure-Python linear (register-machine) GP genome.

A from-scratch reimplementation of the same idea as libs/RedQueenCbind, not a port of it -- this
package exists to validate the Population/FitnessEvaluator/SelectionStrategy/VariationStrategy
abstractions (docs/design/0001) before committing anything to C++ (docs/design/0003 phase 1).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Op = Callable[[float, float], float]


def op_add(a: float, b: float) -> float:
    return a + b


def op_sub(a: float, b: float) -> float:
    return a - b


def op_mul(a: float, b: float) -> float:
    return a * b


def op_div(a: float, b: float) -> float:
    """Protected division: returns 1.0 for a near-zero divisor instead of raising/inf."""
    return a / b if abs(b) > 1e-6 else 1.0


DEFAULT_OPS: tuple[Op, ...] = (op_add, op_sub, op_mul, op_div)


@dataclass(frozen=True, slots=True)
class Instruction:
    op: int  # index into the program's op table
    dst: int  # register index written
    src_a: int  # register index read
    src_b: int  # index into [registers..., inputs...] read (see LinearProgram.run)


@dataclass(frozen=True, slots=True)
class LinearProgram:
    """A linear GP individual: a fixed-length sequence of register-machine instructions."""

    instructions: tuple[Instruction, ...]
    num_registers: int
    num_inputs: int
    ops: tuple[Op, ...] = DEFAULT_OPS

    def run(self, inputs: Sequence[float]) -> list[float]:
        """Executes the program against `inputs`, returning the final register state.

        Reads registers directly rather than copying them into a combined buffer each
        instruction -- that per-instruction copy was a real perf bug fixed in RedQueenCbind's
        C++ version (docs/CODING_GUIDELINES.md).
        """
        registers = [0.0] * self.num_registers
        for instr in self.instructions:
            a = registers[instr.src_a % self.num_registers]
            b_idx = instr.src_b % (self.num_registers + self.num_inputs)
            b = registers[b_idx] if b_idx < self.num_registers else inputs[b_idx - self.num_registers]
            op = self.ops[instr.op % len(self.ops)]
            registers[instr.dst % self.num_registers] = op(a, b)
        return registers

    def output(self, inputs: Sequence[float], register: int = 0) -> float:
        return self.run(inputs)[register]


def random_instruction(
    num_registers: int, num_inputs: int, num_ops: int, rng: random.Random
) -> Instruction:
    return Instruction(
        op=rng.randrange(num_ops),
        dst=rng.randrange(num_registers),
        src_a=rng.randrange(num_registers),
        src_b=rng.randrange(num_registers + num_inputs),
    )


def random_program(
    num_instructions: int,
    num_registers: int,
    num_inputs: int,
    rng: random.Random,
    ops: tuple[Op, ...] = DEFAULT_OPS,
) -> LinearProgram:
    instructions = tuple(
        random_instruction(num_registers, num_inputs, len(ops), rng) for _ in range(num_instructions)
    )
    return LinearProgram(
        instructions=instructions, num_registers=num_registers, num_inputs=num_inputs, ops=ops
    )
