"""Static Checkers strategies -- the fixed opponents every Checkers experiment measures against
(docs/design/0006): a round robin (jobs/checkers_round_robin.py), evolution's fitness opponents
(jobs/checkers_neuro_run.py), the leaderboard's baselines.

Beyond the fixed players, a *trained evaluator* can search: `evaluator(weights, layer_sizes, depth=...)` for
an `evolve.WeightVector`, `graph_evaluator(encoding, depth=...)` for a NEAT genome
(`NeatGenome.graph_encoding()`), both alpha-beta to `depth` plies with the network scoring the leaves --
the classic recipe for learning Checkers (neuroevolution supplies the evaluation, minimax the lookahead).
`material-N` (N >= 3) is the fair opponent for one that searches as deep.

The players live in the Rust game core (rust/core/src/checkers_strategies.rs, docs/design/0009) so
training and the browser run the *same* code: seeded, PCG32 tie-breaks, move for move. This module is
their Python face. Each factory is `(env, rng) -> Strategy`, the env-bound shape
`evolve.MatchFitnessEvaluator(env_aware=True)` expects -- a strategy needs its environment to look
ahead -- and a `Strategy` is `(observation, legal_moves) -> move`. The Rust strategy owns its own PRNG,
seeded from the `random.Random` it is handed, so a game is reproducible from that rng.

`tests/reference_checkers_strategies.py` keeps the original pure-Python implementations as oracles the
Rust ones are checked against.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from games import _native
from games.checkers import Checkers

Strategy = Callable[[list[float], list[tuple]], tuple]
BoundFactory = Callable[[Checkers, random.Random], Strategy]


def _bound(
    name: str,
    weights: Sequence[float] | None = None,
    layer_sizes: Sequence[int] | None = None,
    depth: int = 1,
    graph: Sequence[float] | None = None,
) -> BoundFactory:
    def factory(env: Checkers, rng: random.Random) -> Strategy:
        strategy = _native.CheckersStrategy(
            name,
            rng.getrandbits(63),
            None if weights is None else list(weights),
            None if layer_sizes is None else list(layer_sizes),
            depth,
            None if graph is None else list(graph),
        )
        return lambda observation, moves: env.choose(strategy)

    return factory


random_strategy = _bound("random")
first_legal = _bound("first-legal")
material_1 = _bound("material-1")
material_2 = _bound("material-2")
material_3 = _bound("material-3")
material_4 = _bound("material-4")


def evaluator(weights: Sequence[float], layer_sizes: Sequence[int], depth: int = 1) -> BoundFactory:
    """A trained network as the position evaluator -- an `evolve.WeightVector`'s `weights` and
    `layer_sizes` (32 inputs, 1 output, tanh layers). `depth` 1: play the move whose resulting position,
    seen from the opponent's side, the network likes least *for them*. Deeper: alpha-beta to `depth` plies
    with the network scoring the leaves."""
    return _bound("evaluator", weights, layer_sizes, depth)


def graph_evaluator(encoding: Sequence[float], depth: int = 1) -> BoundFactory:
    """The same for a NEAT genome, given as `NeatGenome.graph_encoding()`."""
    return _bound("evaluator", depth=depth, graph=encoding)


STRATEGIES: dict[str, BoundFactory] = {
    "random": random_strategy,
    "first-legal": first_legal,
    "material-1": material_1,
    "material-2": material_2,
    "material-3": material_3,
    "material-4": material_4,
}
