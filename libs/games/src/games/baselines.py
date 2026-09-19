"""Fixed reference policies per game -- leaderboard anchors (docs/design/0007: a ranking says nothing
without baselines). The policies themselves are in the Rust game core (rust/core/src/baselines.rs), so
the evaluation job and the browser (WASM) run the same code; this module is the registry.

A baseline is registered with the interface it plays under (it reads that observer's encoding) and a
factory `seed -> policy`, where a policy is `observation -> action`. Seeded so evaluation is
reproducible even for the random one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from games import _native

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
    """Uniformly random turn, from the game core's PCG32 stream for `seed` -- the same sequence the
    browser's WASM build produces."""
    return _native.SnakeRandomPolicy(seed)


def _snake_greedy(_seed: int) -> Policy:
    """Turn toward the food unless that's immediately fatal; otherwise any safe move. Reads
    snake/features.v1: danger[0:3] (straight/left/right), heading one-hot[3:7] in RIGHT/DOWN/LEFT/UP
    order, food left/right/up/down[7:11]. Implemented in the game core (baselines.rs)."""
    return lambda observation: _native.snake_greedy_decide(list(observation))


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
