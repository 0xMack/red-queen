# red-queen

A monorepo for exploring reinforcement learning and genetic/evolutionary algorithms via custom,
from-scratch implementations — tested against purpose-built games and simulations, with
interactive visualizations to see what the algorithms are doing internally and make debugging
easier.

## Layout

- `notebooks/` — Jupyter notebooks for exploration, experiments, and write-ups
- `apps/` — front-end applications (e.g. visualizations of games/simulations and algorithm internals)
- `apis/` — backend APIs (e.g. serving simulations/training runs to front-ends)
- `libs/` — shared libraries/packages (algorithm implementations, games/simulations, common utilities)
- `jobs/` — short- or long-running jobs, tasks, and workers (e.g. training runs)

Each top-level directory has its own README describing its contents in more detail as they fill in.

## Current contents

- `libs/redqueen` — a C++/pybind11 linear genetic programming (LGP) implementation (the original
  code this repo started from)
