import type { Snippet } from "~/types/code"

// Teaching a Snake's snippets. Snake's rules live once, in the Rust core (libs/games/rust/core/src/snake.rs); the
// original pure-Python game this chapter was written against is kept as the oracle the Rust is parity-tested
// against (libs/games/tests/reference_snake.py). Both are real code.

export const oldObservation: Snippet = {
  pseudo: `// The original approach: the whole board, flattened, one number per cell
function observe(game)
    grid ← a list of width × height zeros
    for each cell (x, y) holding something
        grid[y × width + x] ← 1 for body, 2 for the head, 3 for food
    return grid`,
  python: {
    source: "libs/games/tests/reference_snake.py",
    code: `# The original approach: the whole board, flattened, one float per cell
def grid_flat(self) -> list[float]:
    grid = [0.0] * (self.width * self.height)
    for x, y, label in self.cells():
        grid[y * self.width + x] = _GRID_VALUES[label]
    return grid`,
  },
  rust: {
    source: "libs/games/rust/core/src/snake.rs",
    code: `// grid-flat.v1: every cell, row-major, empty 0 / body 1 / head 2 / food 3
Observer::GridFlat => {
    let mut grid = vec![0.0; (self.width * self.height) as usize];
    for (x, y, label) in self.cells() {
        grid[(y * self.width + x) as usize] = match label {
            Label::Body => 1.0,
            Label::Head => 2.0,
            Label::Food => 3.0,
        };
    }
    grid
}`,
  },
}

export const newObservation: Snippet = {
  pseudo: `// 11 hand-engineered features instead
function observe(game)
    return [
        danger(straight), danger(left), danger(right),     // would that move kill it?
        heading is right, down, left, up,                  // one-hot: exactly one is 1
        food is left of the head, food is right of it,
        food is above the head, food is below it,
    ]`,
  python: {
    source: "libs/games/tests/reference_snake.py",
    code: `# 11 hand-engineered features instead: danger straight/left/right,
# heading as a one-hot, food direction relative to the head
def features(self) -> list[float]:
    head_x, head_y = self.body[0]
    heading = [1.0 if i == self._direction_index else 0.0 for i in range(4)]
    return [
        self._danger(0), self._danger(-1), self._danger(1),
        *heading,
        *self._food_flags(head_x, head_y),   # food left / right / above / below the head
    ]`,
  },
  rust: {
    source: "libs/games/rust/core/src/snake.rs",
    code: `// features.v1: danger straight/left/right, heading one-hot, food direction
Observer::Features => {
    let (hx, hy) = self.body[0];
    let mut out = vec![self.danger(0), self.danger(-1), self.danger(1)];
    out.extend((0..4).map(|i| if i == self.direction { 1.0 } else { 0.0 }));
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    match self.food {
        Some((fx, fy)) => out.extend([flag(fx < hx), flag(fx > hx), flag(fy < hy), flag(fy > hy)]),
        None => out.extend([0.0; 4]),
    }
    out
}`,
  },
}

export const rewardShaping: Snippet = {
  pseudo: `// Asymmetric on purpose: a symmetric ±0.01 let an evolved policy
// oscillate between two cells forever for ~0 net reward.
new_distance ← |head.x − food.x| + |head.y − food.y|
reward ← +0.01 if new_distance < old_distance, otherwise −0.02`,
  python: {
    source: "libs/games/tests/reference_snake.py",
    code: `# Asymmetric on purpose: a symmetric +/-0.01 let an evolved policy
# oscillate between two cells forever for ~0 net reward.
new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
reward = 0.01 if new_distance < old_distance else -0.02`,
  },
  rust: {
    source: "libs/games/rust/core/src/snake.rs",
    code: `// Asymmetric shaping: closer +0.01, farther -0.02.
let new_distance = (new_head.0 - food.0).abs() + (new_head.1 - food.1).abs();
reward = if new_distance < old_distance { 0.01 } else { -0.02 };`,
  },
}
