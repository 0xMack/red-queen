"""Fixed reference policies per game -- leaderboard anchors (docs/design/0007: a ranking says nothing
without baselines). Lives here, not in jobs/, so the exact same code runs in the evaluation job and
in the browser (the Pyodide worker loads this package to let visitors watch a baseline play).

A baseline is registered with the interface it plays under (it reads that observer's encoding) and a
factory `seed -> policy`, where a policy is `observation -> action`. Seeded so evaluation is
reproducible even for the random one.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Policy = Callable[[list[float]], Any]


@dataclass(frozen=True)
class Baseline:
    game: str
    name: str  # entrant id is f"baseline:{name}"
    label: str
    description: str
    interface: str
    factory: Callable[[int], Policy]

    @property
    def entrant_id(self) -> str:
        return f"baseline:{self.name}"


def _snake_random(seed: int) -> Policy:
    rng = random.Random(seed)
    return lambda _observation: rng.choice((-1, 0, 1))


def _snake_greedy(_seed: int) -> Policy:
    """Turn toward the food unless that's immediately fatal; otherwise any safe move. Reads
    snake/features.v1: danger[0:3] (straight/left/right), heading one-hot[3:7] in RIGHT/DOWN/LEFT/UP
    order, food left/right/up/down[7:11]."""

    def policy(observation: list[float]) -> int:
        danger = {0: observation[0], -1: observation[1], 1: observation[2]}
        heading = observation[3:7].index(1.0)
        food_left, food_right, food_up, food_down = observation[7:11]
        wants = {0: food_right, 1: food_down, 2: food_left, 3: food_up}  # by absolute direction
        for action in (0, -1, 1):
            if not danger[action] and wants[(heading + action) % 4]:
                return action
        for action in (0, -1, 1):
            if not danger[action]:
                return action
        return 0

    return policy


_SNAKE_FEATURES = "snake/features.v1+relative3.v1"

_ALL: list[Baseline] = [
    Baseline(
        game="snake",
        name="random",
        label="Random",
        description="Uniformly random turn every step.",
        interface=_SNAKE_FEATURES,
        factory=_snake_random,
    ),
    Baseline(
        game="snake",
        name="greedy",
        label="Greedy heuristic",
        description="Hand-written rule: head for the food unless that move is immediately fatal.",
        interface=_SNAKE_FEATURES,
        factory=_snake_greedy,
    ),
]


def for_game(game: str) -> list[Baseline]:
    return [b for b in _ALL if b.game == game]


def get(game: str, name: str) -> Baseline:
    for b in _ALL:
        if b.game == game and b.name == name:
            return b
    raise KeyError(f"unknown baseline {name!r} for game {game!r}")
