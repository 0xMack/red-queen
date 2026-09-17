# libs

Shared libraries and packages: algorithm implementations (RL, genetic algorithms, etc.), games and
simulations they run against, and common utilities used across `../apps`, `../apis`, and `../jobs`.

## Contents

- `RedQueenCbind/` — C++/pybind11 linear genetic programming (LGP) implementation
- `evolve/` — pure-Python evolution loop prototype (genome, fitness, selection, variation),
  telemetry-agnostic
- `games/` — toy games/simulations, one module per game (starting with `reach1d`), all
  implementing `evolve`'s `Environment` interface (but not depending on `evolve` itself)
- `telemetry/` — run registry, metrics stream, and artifact store interfaces for observing
  evolving/training populations
