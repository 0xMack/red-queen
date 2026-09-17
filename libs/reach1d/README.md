# reach1d

A toy 1D continuous-control environment: an agent with `(position, velocity)` must reach and stay
near a fixed target by choosing an acceleration each step. Implements
`evolve.simulation.Environment`'s `reset()`/`step()` shape, but doesn't depend on `evolve` at all —
the dependency points the other way (a `SimulationFitnessEvaluator` depends on this environment's
shape, this package doesn't know `evolve` exists), same as `libs/evolve`/`libs/telemetry`.

## Usage

```python
from reach1d import ReachTarget1D, benchmark_environments

env = ReachTarget1D(target=5.0, start_position=0.0)
observation = env.reset()  # (position, velocity)
observation, reward, done = env.step(action=1.0)  # action in [-1, 1]
```

`benchmark_environments()` returns a small fixed set of scenarios (different targets, start
positions, and one with initial velocity) — the "test cases" a
`SimulationFitnessEvaluator` runs one episode against per genome, matching every other
`FitnessEvaluator` in this repo's per-test-case fitness contract.
