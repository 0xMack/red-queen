"""Snake: a toy grid game.

Implements evolve.simulation.Environment's reset()/step() shape (docs/design/0002) -- doesn't
depend on evolve. Also implements games.rendering.Renderable's render_state() so it can be watched
(console, a future frontend, a telemetry trajectory) without touching the fast training
observation at all -- see libs/games/README.md and docs/design/0003's "datasets and games" plan.

Observation: chosen by an observer (docs/design/0007) -- `Snake(observer=...)`; the default,
`SnakeFeatures` ("features.v1", level 3 "engineered"), is 11 hand-engineered features -- danger straight/left/right (would that relative move
hit a wall or body next), current heading as a one-hot (right/down/left/up), and food direction
relative to the head (left/right/up/down), all 0.0/1.0. Replaced an earlier board-size-dependent
flattened-grid observation (100 floats on a 10x10 board): notebooks/0005-neuroevolution-snake.ipynb
trained a real policy against that representation and, after also trying a much bigger population/
generation budget, concluded the ceiling was representational, not a compute shortage (see that
notebook's diversity-vs-plateau analysis). A flat MLP has no spatial prior, so it has to
re-discover "distance to the nearest wall" and "which way is the food" from raw pixel-like input
via evolution alone -- these 11 features hand that structure to the network directly, the standard
representation for small evolved/RL Snake agents. Board-size-independent as a side effect, and
~10x fewer weights for the same hidden-layer width (faster to train and to run). The replaced
representation lives on as `SnakeGridFlat` ("grid-flat.v1", level 1 "full state") rather than being
deleted -- champions trained against it still need it to run, and comparing the two is the point
of docs/design/0007's leaderboards.

Action: -1 (turn left), 0 (go straight), 1 (turn right), relative to the current heading -- so
"reverse into your own neck" is structurally impossible, no special-case masking needed.
`RelativeTurn3` ("relative3.v1") is the matching action adapter for 3-output models.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

# Clockwise order so a "right turn" (action=+1) is simply the next direction and a "left turn"
# (action=-1) the previous one -- no per-direction special-casing needed.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))  # RIGHT, DOWN, LEFT, UP


class Snake:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int = 0,
        max_steps_without_food: int | None = None,
        observer: Any | None = None,
    ):
        # Any games.observation.Observer; SnakeFeatures unless told otherwise, so every pre-0007
        # caller (SimulationFitnessEvaluator, apis/backend sessions, tests) is unchanged.
        self.observer = observer or SnakeFeatures()
        self.width = width
        self.height = height
        self.seed = seed
        self.max_steps_without_food = max_steps_without_food or (width + height) * 4
        self.reset()

    def reset(self) -> list[float]:
        self._rng = random.Random(self.seed)
        mid_x, mid_y = self.width // 2, self.height // 2
        self.body = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]  # head first
        self._direction_index = 0  # RIGHT
        self.score = 0
        self.alive = True
        self._steps_without_food = 0
        self.food = self._place_food()
        return self.observer.encode(self)

    def step(self, action: int) -> tuple[list[float], float, bool]:
        if not self.alive:
            return self.observer.encode(self), 0.0, True

        # int(): a caller across a JSON boundary (apis/backend) can only send a float, never a
        # Python int -- action is conceptually discrete (-1/0/1) regardless of wire type.
        self._direction_index = (self._direction_index + int(action)) % 4
        dx, dy = _DIRECTIONS[self._direction_index]
        head_x, head_y = self.body[0]
        new_head = (head_x + dx, head_y + dy)
        will_eat = new_head == self.food

        out_of_bounds = not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height)
        # the tail cell vacates this move unless the snake is growing, so moving into it is legal
        collision_body = self.body if will_eat else self.body[:-1]
        if out_of_bounds or new_head in collision_body:
            self.alive = False
            return self.observer.encode(self), -1.0, True

        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

        self.body.insert(0, new_head)
        if will_eat:
            self.score += 1
            self._steps_without_food = 0
            self.food = self._place_food()
            reward = 1.0
        else:
            self.body.pop()
            self._steps_without_food += 1
            # shaping: a small nudge toward the food each step, on top of the sparse +1/-1 --
            # +1 alone is too sparse a signal for evolution to bootstrap from on a 10x10 board
            # (validated by running: without shaping, 80 generations only reached "eat one food,
            # then die"). Asymmetric on purpose (validated by running too): a symmetric +/-0.01
            # lets a policy oscillate between two cells forever for ~0 net reward -- a real,
            # lower-risk local optimum than continuing to seek food. Penalizing "farther" more
            # than "closer" is rewarded makes that oscillation net negative, so standing still is
            # worse than food-seeking, not merely no-better.
            new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.01 if new_distance < old_distance else -0.02

        if self._steps_without_food >= self.max_steps_without_food:
            self.alive = False
            return self.observer.encode(self), -1.0, True

        return self.observer.encode(self), reward, False

    def _place_food(self) -> tuple[int, int]:
        empty_cells = [
            (x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.body
        ]
        return self._rng.choice(empty_cells)

    def _cell_labels(self) -> dict[tuple[int, int], str]:
        labels = {cell: "body" for cell in self.body[1:]}
        labels[self.body[0]] = "head"
        labels[self.food] = "food"
        return labels

    def _danger(self, relative_action: int) -> float:
        """Would taking this relative action (-1/0/1) hit a wall or body next step? Body check
        uses body[:-1] (the tail vacates unless growing), same approximation step() uses for the
        non-eating case -- a heuristic for the observation, not the authoritative collision check."""
        direction_index = (self._direction_index + relative_action) % 4
        dx, dy = _DIRECTIONS[direction_index]
        head_x, head_y = self.body[0]
        next_x, next_y = head_x + dx, head_y + dy
        out_of_bounds = not (0 <= next_x < self.width and 0 <= next_y < self.height)
        return 1.0 if out_of_bounds or (next_x, next_y) in self.body[:-1] else 0.0

    def _observation(self) -> list[float]:
        head_x, head_y = self.body[0]
        heading = [1.0 if i == self._direction_index else 0.0 for i in range(4)]
        return [
            self._danger(0),
            self._danger(-1),
            self._danger(1),
            *heading,
            1.0 if self.food[0] < head_x else 0.0,
            1.0 if self.food[0] > head_x else 0.0,
            1.0 if self.food[1] < head_y else 0.0,
            1.0 if self.food[1] > head_y else 0.0,
        ]

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

_HEADING_NAMES = ("→", "↓", "←", "↑")  # _DIRECTIONS order


class SnakeFeatures:
    """Level 3, engineered: the 11 features described in this module's docstring."""

    id = "features.v1"
    level = 3
    description = (
        "11 hand-engineered 0/1 features: danger straight/left/right, heading one-hot, and which "
        "side of the head the food is on. Board-size independent."
    )

    def encode(self, game: Snake) -> list[float]:
        return game._observation()

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


# The pre-0007 flattened-grid encoding, kept exactly (values and row-major order) so champions
# trained against it run unchanged.
_GRID_VALUES = {"body": 1.0, "head": 2.0, "food": 3.0}  # empty = 0.0


class SnakeGridFlat:
    """Level 1, full state: every cell as one float (empty 0, body 1, head 2, food 3), row-major.
    width*height inputs, so board-size dependent. What notebooks/0005 trained against before the
    switch to SnakeFeatures."""

    id = "grid-flat.v1"
    level = 1
    description = (
        "The whole board, one float per cell (empty 0, body 1, head 2, food 3), row-major. "
        "No spatial prior -- the network must discover walls and directions itself."
    )

    def encode(self, game: Snake) -> list[float]:
        grid = [0.0] * (game.width * game.height)
        for (x, y), label in game._cell_labels().items():
            grid[y * game.width + x] = _GRID_VALUES[label]
        return grid

    def feature_names(self, game: Snake) -> list[str]:
        return [f"cell ({x},{y})" for y in range(game.height) for x in range(game.width)]


class RelativeTurn3:
    """3 outputs -> turn left / straight / turn right, by argmax. The convention every Snake policy
    so far was trained under (jobs/snake_neuro_run.py's act(), the Pyodide worker's watch mode)."""

    id = "relative3.v1"
    description = "3 outputs, argmax -> turn left / go straight / turn right, relative to heading."
    num_outputs = 3

    def decode(self, outputs: Sequence[float]) -> int:
        best_index = max(range(len(outputs)), key=lambda i: outputs[i])
        return best_index - 1
