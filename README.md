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
- `docs/` — design documents and cross-cutting write-ups

Each top-level directory has its own README describing its contents in more detail as they fill in.
Contributing (human or agent)? Start with [AGENTS.md](AGENTS.md) — project map, working notes, and
a pointer to [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md).

## Current contents

- `libs/RedQueenCbind` — a C++/pybind11 linear genetic programming (LGP) implementation (the
  original code this repo started from)
- `libs/evolve` — pure-Python evolution loop prototype (genome, fitness, selection, variation) —
  the baseline being validated before anything is ported to C++
- `libs/telemetry` — run registry, metrics stream, and artifact store for observing
  evolving/training populations
- `jobs/baseline_gp_run.py` — runs `libs/evolve` against a fixed benchmark, wired to `telemetry`
  end to end (`uv run python jobs/baseline_gp_run.py`)
- `docs/design/` — numbered design docs: `0001` (GP engine), `0002` (real-time visualization
  architecture), `0003` (algorithm landscape and roadmap)
