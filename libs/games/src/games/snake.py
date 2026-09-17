"""Snake: a toy grid game.

Implements evolve.simulation.Environment's reset()/step() shape (docs/design/0002) -- doesn't
depend on evolve. Also implements games.rendering.Renderable's render_state() so it can be watched
(console, a future frontend, a telemetry trajectory) without touching the fast training
observation at all -- see libs/games/README.md and docs/design/0003's "datasets and games" plan.

Observation: the whole board, flattened row-major, one float per cell (0=empty, 1=body, 2=head,
3=food) -- chosen over a small hand-engineered feature vector so the genome sees the whole board,
at the cost of a larger, board-size-dependent input.

Action: -1 (turn left), 0 (go straight), 1 (turn right), relative to the current heading -- so
"reverse into your own neck" is structurally impossible, no special-case masking needed.
"""

from __future__ import annotations

import random

# Clockwise order so a "right turn" (action=+1) is simply the next direction and a "left turn"
# (action=-1) the previous one -- no per-direction special-casing needed.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))  # RIGHT, DOWN, LEFT, UP

_EMPTY, _BODY, _HEAD, _FOOD = 0.0, 1.0, 2.0, 3.0
_CELL_VALUES = {"body": _BODY, "head": _HEAD, "food": _FOOD}


class Snake:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int = 0,
        max_steps_without_food: int | None = None,
    ):
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
        return self._observation()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        if not self.alive:
            return self._observation(), 0.0, True

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
            return self._observation(), -1.0, True

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
            return self._observation(), -1.0, True

        return self._observation(), reward, False

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

    def _observation(self) -> list[float]:
        grid = [_EMPTY] * (self.width * self.height)
        for (x, y), label in self._cell_labels().items():
            grid[y * self.width + x] = _CELL_VALUES[label]
        return grid

    def render_state(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "cells": self._cell_labels(),
            "score": self.score,
            "alive": self.alive,
        }


def benchmark_environments() -> list[Snake]:
    """A small fixed set of Snake instances, one per seed -- different food sequences, matching
    docs/design/0003's "fixed benchmark problems as an anchor". A small board (10x10) keeps fitness
    evaluation fast across a whole population and many generations."""
    return [Snake(width=10, height=10, seed=seed) for seed in range(5)]
