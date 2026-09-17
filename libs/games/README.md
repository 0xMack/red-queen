# games

Toy games/simulations, one module per game — a single package rather than one `libs/` package per
game (per [docs/design/0003](../../docs/design/0003-algorithm-landscape-and-roadmap.md)'s "datasets
and games" plan), so shared utilities have an obvious home as more games are added instead of being
duplicated per package or living awkwardly in `evolve`.

Every game implements `evolve.simulation.Environment`'s `reset()`/`step()` shape, but this package
never imports `evolve` — the dependency points the other way (a `SimulationFitnessEvaluator`
depends on a game's shape; a game doesn't know `evolve` exists), same as `evolve`/`telemetry`.

Every game also implements a **separate, on-demand `render_state() -> dict`** (`rendering.py`'s
`Renderable`) returning the full board/scene as plain data — not part of `Environment`, and never
called from the fast training path. This is the split that makes "connect it to a frontend/API/
human-play later" cheap: `step()`/`reset()` stay fast and numeric (what a genome trains against);
`render_state()` is what a console renderer, a future Vue3 frontend, or a stored trajectory
artifact would consume, none of which need to touch or slow down the training path.

## Contents

- `reach1d.py` — `ReachTarget1D`: a toy 1D continuous-control task. See
  `notebooks/0004-neuroevolution-reach1d.ipynb`.
- `snake.py` — `Snake`: a grid game. Observation is the whole board, flattened (one float per
  cell); action is a relative turn (left/straight/right), so reversing into your own body is
  structurally impossible. Reward is intentionally asymmetric (penalize moving away from food more
  than moving closer is rewarded) — a symmetric version let an evolved policy oscillate between two
  cells forever for ~0 net reward, a real reward-hacking failure mode found by running it, not
  guessed at in advance.
- `rendering.py` — `Renderable` protocol + `render_grid_ascii()`, a shared ASCII renderer for any
  grid-based game (not Snake-specific) — the kind of reuse consolidating into one package was for.

## Usage

```python
from games.reach1d import ReachTarget1D, benchmark_environments

env = ReachTarget1D(target=5.0, start_position=0.0)
observation = env.reset()
observation, reward, done = env.step(action=1.0)
```

```python
from games.rendering import render_grid_ascii
from games.snake import Snake

env = Snake(width=10, height=10, seed=0)
env.reset()
state = env.render_state()
print(render_grid_ascii(state["width"], state["height"], state["cells"], {"head": "@", "body": "o", "food": "*"}))
```
