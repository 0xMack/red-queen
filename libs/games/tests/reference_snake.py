"""The oracle for the Rust port (docs/design/0009 Decision 2): Snake as it was in pure Python before the
game core moved to Rust, changed in exactly two ways: food placement draws from an independent
pure-Python PCG32 (the core's specified PRNG) instead of Mersenne Twister, and filling the board ends
the game as a win (the original crashed choosing food from no empty cells). Given the same seed, this
and `games.snake.Snake` must produce identical games; `test_native_parity.py` checks that they do.

Test-only on purpose: it's the executable spec the Rust is held to, not a second implementation for
anything to run.
"""

from __future__ import annotations

from collections import deque

_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
STREAM = 54
_MULTIPLIER = 6364136223846793005


class Pcg32:
    """PCG32 (XSH-RR, 64-bit state), seeded as pcg32_srandom_r(seed, 54) -- see rust/core/src/pcg.rs."""

    def __init__(self, seed: int):
        self.state = 0
        self.inc = ((STREAM << 1) | 1) & _MASK64
        self.next_u32()
        self.state = (self.state + seed) & _MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * _MULTIPLIER + self.inc) & _MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & _MASK32
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _MASK32

    def bounded(self, n: int) -> int:
        threshold = ((1 << 32) - n) % n
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % n


_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_GRID_VALUES = {"body": 1.0, "head": 2.0, "food": 3.0}


class ReferenceSnake:
    def __init__(self, width: int = 10, height: int = 10, seed: int = 0, max_steps_without_food: int | None = None):
        self.width = width
        self.height = height
        self.seed = seed
        self.max_steps_without_food = max_steps_without_food or (width + height) * 4
        self.reset()

    def reset(self) -> None:
        self._rng = Pcg32(self.seed)
        mid_x, mid_y = self.width // 2, self.height // 2
        self.body = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self._direction_index = 0
        self.score = 0
        self.alive = True
        self._steps_without_food = 0
        self.food = self._place_food()

    def step(self, action: int) -> tuple[float, bool]:
        if not self.alive:
            return 0.0, True
        self._direction_index = (self._direction_index + int(action)) % 4
        dx, dy = _DIRECTIONS[self._direction_index]
        head_x, head_y = self.body[0]
        new_head = (head_x + dx, head_y + dy)
        will_eat = new_head == self.food
        out_of_bounds = not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height)
        collision_body = self.body if will_eat else self.body[:-1]
        if out_of_bounds or new_head in collision_body:
            self.alive = False
            return -1.0, True
        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        self.body.insert(0, new_head)
        if will_eat:
            self.score += 1
            self._steps_without_food = 0
            self.food = self._place_food()
            reward = 1.0
            if self.food is None:
                self.alive = False
                return reward, True
        else:
            self.body.pop()
            self._steps_without_food += 1
            new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.01 if new_distance < old_distance else -0.02
        if self._steps_without_food >= self.max_steps_without_food:
            self.alive = False
            return -1.0, True
        return reward, False

    def _place_food(self) -> tuple[int, int] | None:
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.body]
        return empty_cells[self._rng.bounded(len(empty_cells))] if empty_cells else None

    def cells(self) -> list[tuple[int, int, str]]:
        labels = {cell: "body" for cell in self.body[1:]}
        labels[self.body[0]] = "head"
        if self.food is not None:
            labels[self.food] = "food"
        return [(x, y, label) for (x, y), label in labels.items()]

    def _danger(self, relative_action: int) -> float:
        direction_index = (self._direction_index + relative_action) % 4
        dx, dy = _DIRECTIONS[direction_index]
        head_x, head_y = self.body[0]
        next_x, next_y = head_x + dx, head_y + dy
        out_of_bounds = not (0 <= next_x < self.width and 0 <= next_y < self.height)
        return 1.0 if out_of_bounds or (next_x, next_y) in self.body[:-1] else 0.0

    def features(self) -> list[float]:
        head_x, head_y = self.body[0]
        heading = [1.0 if i == self._direction_index else 0.0 for i in range(4)]
        return [
            self._danger(0),
            self._danger(-1),
            self._danger(1),
            *heading,
            *(self._food_flags(head_x, head_y)),
        ]

    def _food_flags(self, head_x: int, head_y: int) -> list[float]:
        if self.food is None:
            return [0.0] * 4
        return [
            1.0 if self.food[0] < head_x else 0.0,
            1.0 if self.food[0] > head_x else 0.0,
            1.0 if self.food[1] < head_y else 0.0,
            1.0 if self.food[1] > head_y else 0.0,
        ]

    EGO_RAYS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, -1), (-1, 1))  # (ahead, right)

    def egocentric(self) -> list[float]:
        """`egocentric.v1`, written from the spec rather than ported: what a ray sees is found by walking it and
        asking, at each step, which of the *original* body segments still exist after that many moves (the tail
        vacates one segment per move, so the segments left are the first `len - moves`)."""
        head_x, head_y = self.body[0]
        forward = ((1, 0), (0, 1), (-1, 0), (0, -1))[self._direction_index]
        right = ((1, 0), (0, 1), (-1, 0), (0, -1))[(self._direction_index + 1) % 4]
        out: list[float] = []
        for ahead, side in self.EGO_RAYS:
            step = (ahead * forward[0] + side * right[0], ahead * forward[1] + side * right[1])
            moves = abs(ahead) + abs(side)
            wall = body = food = 0.0
            k = 1
            while True:
                x, y = head_x + k * step[0], head_y + k * step[1]
                if not (0 <= x < self.width and 0 <= y < self.height):
                    wall = 1.0 / k
                    break
                remaining = self.body[: max(len(self.body) - k * moves, 0)]
                if not body and (x, y) in remaining:
                    body = 1.0 / k
                elif not body and not food and (x, y) == self.food:
                    food = 1.0 / k
                k += 1
            out += [wall, body, food]

        scale = max(max(self.width, self.height) - 1, 1)

        def offset(cell):
            dx, dy = cell[0] - head_x, cell[1] - head_y
            return [(dx * forward[0] + dy * forward[1]) / scale, (dx * right[0] + dy * right[1]) / scale]

        out += offset(self.food) if self.food is not None else [0.0, 0.0]
        out += offset(self.body[-1])
        out += [self.score / (self.width * self.height), self._steps_without_food / self.max_steps_without_food]
        return out

    def egocentric_v2(self) -> list[float]:
        """`egocentric.v2` from its spec: `egocentric.v1`, then per move (left, straight, right) the share of free
        cells reachable after it and whether the tail is reachable. Breadth-first over sets -- not the Rust's
        stack-and-arrays fill -- so the two can only agree by both being right."""
        spaces, tails = [], []
        for relative in (-1, 0, 1):
            dx, dy = _DIRECTIONS[(self._direction_index + relative) % 4]
            head = (self.body[0][0] + dx, self.body[0][1] + dy)
            eats = head == self.food
            body_after = [head, *(self.body if eats else self.body[:-1])]
            inside = 0 <= head[0] < self.width and 0 <= head[1] < self.height
            if not inside or head in body_after[1:]:
                spaces.append(0.0)
                tails.append(0.0)
                continue
            occupied = set(body_after)
            tail = body_after[-1]
            reached: set[tuple[int, int]] = set()
            frontier = deque([head])
            while frontier:
                x, y = frontier.popleft()
                for ddx, ddy in _DIRECTIONS:
                    cell = (x + ddx, y + ddy)
                    if 0 <= cell[0] < self.width and 0 <= cell[1] < self.height and cell not in occupied | reached:
                        reached.add(cell)
                        frontier.append(cell)
            free = self.width * self.height - len(body_after)
            spaces.append(len(reached) / free if free else 0.0)
            touching = {(tail[0] + ddx, tail[1] + ddy) for ddx, ddy in _DIRECTIONS}
            tails.append(1.0 if tail == head or touching & (reached | {head}) else 0.0)
        return self.egocentric() + spaces + tails

    def grid_onehot(self) -> list[float]:
        out = [0.0] * (3 * self.width * self.height + 4)
        channel = {"body": 0, "head": 1, "food": 2}
        for x, y, label in self.cells():
            out[3 * (y * self.width + x) + channel[label]] = 1.0
        out[3 * self.width * self.height + self._direction_index] = 1.0
        return out

    def grid_flat(self) -> list[float]:
        grid = [0.0] * (self.width * self.height)
        for x, y, label in self.cells():
            grid[y * self.width + x] = _GRID_VALUES[label]
        return grid


def reference_greedy(observation: list[float]) -> int:
    """games/baselines.py's greedy policy as it was in Python."""
    danger = {0: observation[0], -1: observation[1], 1: observation[2]}
    heading = observation[3:7].index(1.0)
    food_left, food_right, food_up, food_down = observation[7:11]
    wants = {0: food_right, 1: food_down, 2: food_left, 3: food_up}
    for action in (0, -1, 1):
        if not danger[action] and wants[(heading + action) % 4]:
            return action
    for action in (0, -1, 1):
        if not danger[action]:
            return action
    return 0
