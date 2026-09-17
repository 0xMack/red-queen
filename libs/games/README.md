# games

Toy games/simulations, one module per game — a single package rather than one `libs/` package per
game (per [docs/design/0003](../../docs/design/0003-algorithm-landscape-and-roadmap.md)'s "datasets
and games" plan), so shared utilities have an obvious home as more games are added instead of being
duplicated per package or living awkwardly in `evolve`.

Every game implements `evolve.simulation.Environment`'s `reset()`/`step()` shape, but this package
never imports `evolve` — the dependency points the other way (a `SimulationFitnessEvaluator`
depends on a game's shape; a game doesn't know `evolve` exists), same as `evolve`/`telemetry`.

## Contents

- `reach1d.py` — `ReachTarget1D`: a toy 1D continuous-control task. See
  `notebooks/0004-neuroevolution-reach1d.ipynb`.

## Usage

```python
from games.reach1d import ReachTarget1D, benchmark_environments

env = ReachTarget1D(target=5.0, start_position=0.0)
observation = env.reset()
observation, reward, done = env.step(action=1.0)
```
