"""A pure-Python tree GP genome -- the second representation (docs/design/0003 phase 4), added to
prove `Population`/`SelectionStrategy`/`VariationStrategy` actually stay generic across genome
types, not just across selection strategies on `LinearProgram`.

Classic Koza-style expression trees: a `TreeProgram` is a binary tree of `FunctionNode`s (an op
from `evolve.genome.DEFAULT_OPS` applied to two children) over `Terminal` leaves (either the input
variable `x`, or an ephemeral random constant). Unlike `LinearProgram`, tree size genuinely varies
by construction (no fixed instruction count) -- `node_count()` plays the same role
`effective_instruction_count()` does for linear GP: the complexity objective `ParetoSelection`
plugs in via its injected `complexity` callable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from evolve.genome import DEFAULT_OPS, Op


@dataclass(frozen=True, slots=True)
class Terminal:
    value: float | None  # None means "the input variable x"; otherwise an ephemeral random constant

    def evaluate(self, x: float, ops: tuple[Op, ...] = DEFAULT_OPS) -> float:
        # `ops` is unused (a Terminal has no operator to apply) -- accepted only so Terminal and
        # FunctionNode share one call signature, since FunctionNode.evaluate calls into either
        # child without knowing which kind it is.
        return x if self.value is None else self.value


@dataclass(frozen=True, slots=True)
class FunctionNode:
    op: int  # index into `ops`
    left: Node
    right: Node

    def evaluate(self, x: float, ops: tuple[Op, ...] = DEFAULT_OPS) -> float:
        return ops[self.op % len(ops)](self.left.evaluate(x), self.right.evaluate(x))


Node = Terminal | FunctionNode


@dataclass(frozen=True, slots=True)
class TreeProgram:
    """A tree GP individual."""

    root: Node
    ops: tuple[Op, ...] = DEFAULT_OPS

    def evaluate(self, x: float) -> float:
        return self.root.evaluate(x, self.ops)

    def node_count(self) -> int:
        return len(_flatten(self.root))

    def depth(self) -> int:
        return _depth(self.root)


def _flatten(node: Node) -> list[Node]:
    """Pre-order node list: the node itself, then its left subtree, then its right subtree."""
    if isinstance(node, FunctionNode):
        return [node, *_flatten(node.left), *_flatten(node.right)]
    return [node]


def _depth(node: Node) -> int:
    if isinstance(node, FunctionNode):
        return 1 + max(_depth(node.left), _depth(node.right))
    return 0


def _replace_at(node: Node, index: int, replacement: Node) -> Node:
    """Rebuilds `node`'s tree with the pre-order `index`-th node replaced by `replacement`."""
    result, _ = _replace_at_helper(node, index, replacement, counter=[0])
    return result


def _replace_at_helper(node: Node, index: int, replacement: Node, counter: list[int]) -> tuple[Node, bool]:
    current = counter[0]
    counter[0] += 1
    if current == index:
        return replacement, True
    if isinstance(node, FunctionNode):
        new_left, done = _replace_at_helper(node.left, index, replacement, counter)
        if done:
            return replace(node, left=new_left), True
        new_right, done = _replace_at_helper(node.right, index, replacement, counter)
        if done:
            return replace(node, right=new_right), True
    return node, False


def random_node(max_depth: int, rng: random.Random, terminal_prob: float = 0.3) -> Node:
    if max_depth <= 0 or rng.random() < terminal_prob:
        if rng.random() < 0.5:
            return Terminal(value=None)
        return Terminal(value=rng.uniform(-2.0, 2.0))
    return FunctionNode(
        op=rng.randrange(len(DEFAULT_OPS)),
        left=random_node(max_depth - 1, rng, terminal_prob),
        right=random_node(max_depth - 1, rng, terminal_prob),
    )


def random_tree_program(max_depth: int, rng: random.Random) -> TreeProgram:
    return TreeProgram(root=random_node(max_depth, rng))


class TreeCrossoverMutation:
    """Subtree crossover between two parents, then (with `mutation_rate` probability) subtree
    mutation, both standard tree GP operators. `max_tree_depth` is a depth limit -- a standard,
    common mitigation for tree GP "bloat" (crossover can otherwise grow trees without bound); an
    offspring that would exceed it is discarded in favor of an unmodified copy of parent `a`.
    """

    def __init__(self, mutation_rate: float = 0.1, max_mutation_depth: int = 3, max_tree_depth: int = 8):
        self._mutation_rate = mutation_rate
        self._max_mutation_depth = max_mutation_depth
        self._max_tree_depth = max_tree_depth

    def vary(self, parents: list[TreeProgram], rng: random.Random) -> TreeProgram:
        a = parents[0]
        b = parents[1] if len(parents) > 1 else parents[0]

        a_nodes = _flatten(a.root)
        b_nodes = _flatten(b.root)
        crossover_point = rng.randrange(len(a_nodes))
        donor = b_nodes[rng.randrange(len(b_nodes))]
        child_root = _replace_at(a.root, crossover_point, donor)

        if rng.random() < self._mutation_rate:
            mutation_nodes = _flatten(child_root)
            mutation_point = rng.randrange(len(mutation_nodes))
            new_subtree = random_node(self._max_mutation_depth, rng)
            child_root = _replace_at(child_root, mutation_point, new_subtree)

        if _depth(child_root) > self._max_tree_depth:
            return a

        return replace(a, root=child_root)
