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
- `libs/tinylm` — a small transformer LM, built on `libs/autodiff` — character-level, causal
  self-attention, trained by gradients (not evolution) on *Alice's Adventures in Wonderland*
- `jobs/baseline_gp_run.py` — runs `libs/evolve` against a fixed benchmark, wired to `telemetry`
  end to end (`uv run python jobs/baseline_gp_run.py`); `jobs/snake_neuro_run.py` does the same for
  neuroevolution against `games.snake`, writing champions as real, round-trippable
  `WeightVector.to_json()` artifacts; `jobs/control.py` adds pause/resume support via a plain
  `RunStatus` check, the job-side half of the control API below
- `apis/backend` — a FastAPI service exposing `libs/telemetry` (runs, metrics history, a live SSE
  metrics stream, artifacts, and pause/resume/step control) and `libs/games` (server-side game
  sessions: create, step, trajectory) — see `apis/backend/README.md`.
- `apps/frontend` — a Nuxt 4 app: a run list and a live run-detail view (SSE-backed chart) over
  `apis/backend`, plus two ways to watch a game entirely client-side via Pyodide in a Web Worker (a
  real CPython-in-WASM runtime running `libs/games`'/`libs/evolve`'s actual source, off the main
  thread, zero backend round trips per tick): `/play/snake` (a human plays) and `/watch/{runId}` (a
  trained policy plays, live-following a still-training run's current-best champion or replaying a
  finished run's) — see `apis/backend/README.md` to run the API it depends on, and
  `apps/frontend/README.md` for this app.
- `notebooks/` — algorithm comparisons: `0001` (tournament vs. lexicase selection), `0002`
  (Pareto selection, accuracy vs. program size), `0003` (linear vs. tree genome representation),
  `0004` (neuroevolution on reach1d), `0005` (neuroevolution on Snake — an honest, modest result),
  `0006` (a transformer LM trained entirely from scratch, first gradient-trained thing in this repo)
- `docs/design/` — numbered design docs: `0001` (GP engine), `0002` (real-time visualization
  architecture), `0003` (algorithm landscape and roadmap), `0004` (small transformer/LM from
  scratch), `0005` (frontend + API contracts/endpoint definitions)
