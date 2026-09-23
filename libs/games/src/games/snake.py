"""Snake: a toy grid game.

The rules, observers and action adapter run in the Rust game core (`libs/games/rust/core/src/snake.rs`,
docs/design/0009): the same implementation the browser runs as WebAssembly, so a game a visitor watches
is exactly the game that was trained on and evaluated. This module is its Python face -- the API every
caller already used (`Snake(observer=...)`, `reset()`/`step()`, `render_state()`, `body`/`food`) -- plus
the observer/adapter descriptors that name those encodings for the interface registry.

Implements evolve.simulation.Environment's reset()/step() shape (docs/design/0002) -- doesn't depend
on evolve -- and games.rendering.Renderable's render_state().

Observation: chosen by an observer (docs/design/0007) -- `Snake(observer=...)`; the default,
`SnakeFeatures` ("features.v1", level 3 "engineered"), is 11 hand-engineered features -- danger
straight/left/right (would that relative move hit a wall or body next), current heading as a one-hot
(right/down/left/up), and food direction relative to the head (left/right/up/down), all 0.0/1.0.
Replaced an earlier board-size-dependent flattened-grid observation (100 floats on a 10x10 board):
notebooks/0005-neuroevolution-snake.ipynb trained a real policy against that representation and,
after also trying a much bigger population/generation budget, concluded the ceiling was
representational, not a compute shortage. A flat MLP has no spatial prior, so it has to re-discover
"distance to the nearest wall" and "which way is the food" from raw pixel-like input via evolution
alone -- these 11 features hand that structure to the network directly. The replaced representation
lives on as `SnakeGridFlat` ("grid-flat.v1", level 1 "full state"), so champions trained against it
still run.

Rules worth knowing (all in snake.rs): moving into the tail's cell is legal (it vacates) unless the
snake is growing; reward is +1 for food, -1 for dying or starving (no food within
`max_steps_without_food`, default `(width + height) * 4`), and an *asymmetric* shaping nudge
otherwise (+0.01 closer, -0.02 farther: a symmetric one let evolved policies oscillate between two
cells forever -- docs/CODING_GUIDELINES.md). Food placement uses the core's specified PRNG (PCG32),
not Python's `random`, so a seed means the same game in Python and in the browser -- and a different
game than it did before the port (hence protocol `snake.score.v2`).

Action: -1 (turn left), 0 (go straight), 1 (turn right), relative to the current heading -- so
"reverse into your own neck" is structurally impossible, no special-case masking needed.
`RelativeTurn3` ("relative3.v1") is the matching action adapter for 3-output models.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from games import _native

# Clockwise order (RIGHT, DOWN, LEFT, UP): a right turn (+1) is the next direction, a left turn (-1)
# the previous one. Mirrors snake.rs's DIRECTIONS.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Snake:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int = 0,
        max_steps_without_food: int | None = None,
        observer: Any | None = None,
    ):
        # Any games.observation.Observer; SnakeFeatures unless told otherwise. An observer the core
        # implements (`native_id`) is encoded in the same native call as the step; any other observer
        # reads the public state (body/food/heading) itself.
        self.observer = observer or SnakeFeatures()
        self.width = width
        self.height = height
        self.seed = seed
        self.max_steps_without_food = max_steps_without_food or (width + height) * 4
        self._native_observer: str | None = getattr(self.observer, "native_id", None)
        self._core = _native.SnakeCore(width, height, seed, self.max_steps_without_food)

    def __copy__(self):
        # The state lives in the native core, so even a "shallow" copy must not share it.
        return copy.deepcopy(self)

    def reset(self) -> list[float]:
        self._core.reset()
        return self.observer.encode(self)

    def step(self, action: int) -> tuple[list[float], float, bool]:
        # int(): a caller across a JSON boundary (apis/backend) can only send a float, never a
        # Python int -- action is conceptually discrete (-1/0/1) regardless of wire type.
        if self._native_observer is not None:
            return self._core.step_encode(int(action), self._native_observer)
        reward, done = self._core.step(int(action))
        return self.observer.encode(self), reward, done

    def play(self, policy: _native.Policy, max_steps: int) -> tuple[float, int]:
        """One whole episode from a fresh reset, `policy` (`games.nets.native_policy`) choosing every move
        through this game's observer and `relative3.v1`, for at most `max_steps` steps: (total reward, steps).
        The same loop as stepping from Python -- the same moves and total, bit for bit -- in one native call."""
        if self._native_observer is None:
            raise ValueError(f"{self.observer.id} isn't implemented in the game core, so it can't play natively")
        return self._core.play(policy, self._native_observer, max_steps)

    # --- State (read by observers, renderers, tests; settable to set up a scenario) -----------------

    @property
    def body(self) -> list[tuple[int, int]]:
        """Head first."""
        return self._core.body

    @body.setter
    def body(self, cells: Sequence[tuple[int, int]]) -> None:
        self._core.set_state([tuple(c) for c in cells], self._core.direction, self._core.food)

    @property
    def food(self) -> tuple[int, int] | None:
        return self._core.food

    @food.setter
    def food(self, cell: tuple[int, int] | None) -> None:
        self._core.set_state(self._core.body, self._core.direction, None if cell is None else tuple(cell))

    @property
    def _direction_index(self) -> int:
        return self._core.direction

    @_direction_index.setter
    def _direction_index(self, index: int) -> None:
        self._core.set_state(self._core.body, index, self._core.food)

    @property
    def score(self) -> int:
        return self._core.score

    @property
    def alive(self) -> bool:
        return self._core.alive

    def _observation(self) -> list[float]:
        """The features.v1 encoding, whatever this game's observer is (kept for callers from before
        observers were pluggable)."""
        return self._core.encode(SnakeFeatures.native_id)

    def _cell_labels(self) -> dict[tuple[int, int], str]:
        """Body behind the head (nearest the head first), then head, then food -- the order
        renderers rely on (see docs/CODING_GUIDELINES.md on Snake's render order)."""
        return {(x, y): label for x, y, label in self._core.cells()}

    def render_state(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "cells": self._cell_labels(),
            "score": self.score,
            "alive": self.alive,
        }


BENCHMARK_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def benchmark_environments(observer: Any | None = None) -> list[Snake]:
    """A small fixed set of Snake instances, one per seed -- different food sequences, matching
    docs/design/0003's "fixed benchmark problems as an anchor". A small board (10x10) keeps fitness
    evaluation fast across a whole population and many generations. These are *training* seeds;
    leaderboard evaluation (docs/design/0007) uses held-out ones."""
    return [Snake(width=10, height=10, seed=seed, observer=observer) for seed in BENCHMARK_SEEDS]


# --- Observers and action adapters (docs/design/0007) ----------------------------------------------
# Descriptors: the encodings themselves live in snake.rs (`native_id` names them there), so Python and
# the browser can't drift apart.

_HEADING_NAMES = ("→", "↓", "←", "↑")  # _DIRECTIONS order


class SnakeFeatures:
    """Level 3, engineered: the 11 features described in this module's docstring."""

    id = "features.v1"
    native_id = "features.v1"
    level = 3
    description = (
        "11 hand-engineered 0/1 features: danger straight/left/right, heading one-hot, and which "
        "side of the head the food is on. Board-size independent."
    )

    def encode(self, game: Snake) -> list[float]:
        return game._core.encode(self.native_id)

    def feature_names(self, game: Snake) -> list[str]:
        return [
            "danger ahead",
            "danger left",
            "danger right",
            *(f"heading {h}" for h in _HEADING_NAMES),
            "food ←",
            "food →",
            "food ↑",
            "food ↓",
        ]


_RAY_NAMES = ("left", "front-left", "front", "front-right", "right", "back-left", "back-right")


class SnakeEgocentric:
    """Level 2, local / egocentric: what the head can *see*, in its own frame (docs/design/0007's L2, which Snake
    had no observer for until now). 7 line-of-sight rays fan out from the head (left, front-left, front,
    front-right, right, back-left, back-right; straight back is always the neck) and each reports three
    proximities, `1 / distance` (0 = not seen): the wall, the first body segment that will still be there when the
    head arrives, and food (hidden behind body). Then the food and the tail as (ahead, right) offsets from the head,
    apples eaten (score / board cells) and the hunger clock (steps since food / starvation limit). No absolute
    heading, because every value is already relative to it. 27 values, board-size independent, and cheap:
    O(rays x board side) per step. A ray at distance 1 is exactly `features.v1`'s danger."""

    id = "egocentric.v1"
    native_id = "egocentric.v1"
    level = 2
    description = (
        "7 line-of-sight rays from the head (wall / body / food proximity each), plus the food and tail as "
        "ahead/right offsets, apples eaten and a hunger clock. Everything in the head's frame."
    )

    def encode(self, game: Snake) -> list[float]:
        return game._core.encode(self.native_id)

    def feature_names(self, game: Snake) -> list[str]:
        return [
            *(f"{ray} {kind}" for ray in _RAY_NAMES for kind in ("wall", "body", "food")),
            "food ahead",
            "food right",
            "tail ahead",
            "tail right",
            "apples eaten",
            "hunger",
        ]


class SnakeGridFlat:
    """Level 1, full state: every cell as one float (empty 0, body 1, head 2, food 3), row-major.
    width*height inputs, so board-size dependent. What notebooks/0005 trained against before the
    switch to SnakeFeatures -- kept exactly (values and order) so champions trained on it still run."""

    id = "grid-flat.v1"
    native_id = "grid-flat.v1"
    level = 1
    description = (
        "The whole board, one float per cell (empty 0, body 1, head 2, food 3), row-major. "
        "No spatial prior -- the network must discover walls and directions itself."
    )

    def encode(self, game: Snake) -> list[float]:
        return game._core.encode(self.native_id)

    def feature_names(self, game: Snake) -> list[str]:
        return [f"cell ({x},{y})" for y in range(game.height) for x in range(game.width)]


class RelativeTurn3:
    """3 outputs -> turn left / straight / turn right, by argmax (first maximum wins). The convention
    every Snake policy so far was trained under."""

    id = "relative3.v1"
    description = "3 outputs, argmax -> turn left / go straight / turn right, relative to heading."
    num_outputs = 3

    def decode(self, outputs: Sequence[float]) -> int:
        return _native.relative3_decode(list(outputs))
