"""The original pure-Python Checkers strategies' *scoring*, kept as oracles for the Rust ones
(rust/core/src/checkers_strategies.rs) -- the same role reference_checkers.py plays for the rules.

Each function returns one score per legal move (higher is better); the Rust strategy must pick a move
with the top score. Tie-breaking is deliberately not compared: the Rust side breaks ties with its own
PCG32, uniformly, so the oracle can only say which moves are *acceptable*.
"""

from __future__ import annotations

import copy
import math

from games.checkers import Checkers

WIN_SCORE = 1000.0


def material_1_scores(env: Checkers) -> list[float]:
    # simulate() encodes the position from the opponent's side, so the mover's material is the negated sum.
    return [-sum(env.simulate(m)) for m in env.legal_moves()]


def material_2_scores(env: Checkers) -> list[float]:
    me = env.current_player()
    scores = []
    for move in env.legal_moves():
        child = copy.deepcopy(env)
        _, _, done = child.step(move)
        if done:
            scores.append(WIN_SCORE if child.winner() == me else 0.0)
        else:
            # After the opponent's reply it's my move again, so simulate() is from my perspective.
            scores.append(min(sum(child.simulate(reply)) for reply in child.legal_moves()))
    return scores


def forward(weights, layer_sizes, observation) -> float:
    """evolve.neuro.WeightVector's forward pass: per layer `out*in` weights then `out` biases, tanh."""
    activations = list(observation)
    offset = 0
    for n_in, n_out in zip(layer_sizes, layer_sizes[1:]):
        nxt = []
        for o in range(n_out):
            total = weights[offset + n_in * n_out + o]
            for k in range(n_in):
                total += weights[offset + o * n_in + k] * activations[k]
            nxt.append(math.tanh(total))
        offset += n_out * (n_in + 1)
        activations = nxt
    return activations[0]


def evaluator_scores(env: Checkers, weights, layer_sizes) -> list[float]:
    return [-forward(weights, layer_sizes, env.simulate(m)) for m in env.legal_moves()]
