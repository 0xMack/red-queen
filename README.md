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
- `libs/autodiff` — reverse-mode automatic differentiation, built from scratch (`Tensor`,
  NumPy-array-valued, not scalar-valued) — the foundation for `libs/tinylm`'s transformer
- `libs/evolve` — pure-Python evolution loop prototype (genome, fitness, selection, variation) —
  the baseline being validated before anything is ported to C++
- `libs/games` — toy games/simulations, one module per game (`reach1d`, a 1D continuous-control
  environment; `snake`, a grid game), each also exposing a `render_state()` decoupled from the
  fast training path — see `libs/games/README.md`
- `libs/telemetry` — run registry, metrics stream, and artifact store for observing
  evolving/training populations
- `jobs/baseline_gp_run.py` — runs `libs/evolve` against a fixed benchmark, wired to `telemetry`
  end to end (`uv run python jobs/baseline_gp_run.py`)
- `notebooks/` — algorithm comparisons: `0001` (tournament vs. lexicase selection), `0002`
  (Pareto selection, accuracy vs. program size), `0003` (linear vs. tree genome representation),
  `0004` (neuroevolution on reach1d), `0005` (neuroevolution on Snake — an honest, modest result)
- `docs/design/` — numbered design docs: `0001` (GP engine), `0002` (real-time visualization
  architecture), `0003` (algorithm landscape and roadmap), `0004` (small transformer/LM from
  scratch)
