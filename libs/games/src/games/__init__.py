"""Toy games/simulations, one module per game, all implementing evolve.simulation.Environment's
reset()/step() shape without depending on evolve itself (dependency points the other way, same
as evolve/telemetry). One package rather than one-per-game so shared utilities (grid helpers,
reward-shaping helpers, etc.) have an obvious home as more games are added, instead of being
duplicated per package or awkwardly living in evolve.

Import a specific game from its module, e.g. `from games.reach1d import ReachTarget1D` or
`from games import reach1d; reach1d.benchmark_environments()` -- this top-level package
deliberately doesn't re-export every game's symbols flat, since different games are likely to
reuse names like `benchmark_environments()`.
"""
