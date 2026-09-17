# libs

Shared libraries and packages: algorithm implementations (RL, genetic algorithms, etc.), games and
simulations they run against, and common utilities used across `../apps`, `../apis`, and `../jobs`.

## Contents

- `RedQueenCbind/` — C++/pybind11 linear genetic programming (LGP) implementation
- `evolve/` — pure-Python evolution loop prototype (genome, fitness, selection, variation),
  telemetry-agnostic
- `reach1d/` — a toy 1D continuous-control environment, implementing `evolve`'s `Environment`
  interface (but not depending on `evolve` itself)
- `telemetry/` — run registry, metrics stream, and artifact store interfaces for observing
  evolving/training populations
