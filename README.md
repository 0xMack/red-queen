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
  the baseline being validated before anything is ported to C++. `match.py` is the two-player
  sibling of the single-agent `simulation.py`: `MultiAgentEnvironment`, `play_match()` (pit any
  strategy against any strategy — static heuristic, evolved genome, classifier, all the same
  `(observation, legal_moves) -> move` shape), and `MatchFitnessEvaluator` (fitness from match
  outcomes against reference opponents)
- `libs/games` — toy games/simulations, one module per game (`reach1d`, a 1D continuous-control
  environment; `snake`, a grid game — observation is 11 hand-engineered features, not a raw
  flattened grid, after a retrained result confirmed representation was the ceiling, not compute;
  `checkers`, the first two-player game — real rules including mandatory captures/multi-jump
  chains, reuses `render_state()`'s exact shape with just a wider piece-label vocabulary), each
  also exposing a `render_state()` decoupled from the fast training path — see
  `libs/games/README.md`
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
  `apis/backend`, game pages where visitors watch trained champions and baselines play and then play
  themselves, a Learn section, and TinyLM generating text -- all inference client-side
  (docs/design/0009): the Rust game core as WebAssembly plus model packages in ONNX Runtime Web
  (WebGPU or WASM, whichever the device supports, with a plain-language reason when neither fits), so
  a visitor watching costs the server nothing per move. See `apps/frontend/README.md`.
- `notebooks/` — algorithm comparisons: `0001` (tournament vs. lexicase selection), `0002`
  (Pareto selection, accuracy vs. program size), `0003` (linear vs. tree genome representation),
  `0004` (neuroevolution on reach1d), `0005` (neuroevolution on Snake, with the original flattened-
  grid observation — an honest, modest result later revisited: `jobs/snake_neuro_run.py`'s
  hand-engineered-feature observation trains a dramatically stronger policy on the same
  generation-class budget), `0006` (a transformer LM trained entirely from scratch, first
  gradient-trained thing in this repo)
- `docs/design/` — numbered design docs: `0001` (GP engine), `0002` (real-time visualization
  architecture), `0003` (algorithm landscape and roadmap), `0004` (small transformer/LM from
  scratch), `0005` (frontend + API contracts/endpoint definitions), `0006` (multi-agent games and
  the strategy/match framework, checkers as the first exercise of it — done: the framework, the game,
  an evolved position-evaluator champion, and a playable page (human vs. bot, bot vs. bot, an arena
  that measures strategies), all client-side)
