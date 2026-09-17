# AGENTS.md

Monorepo for exploring reinforcement learning and genetic/evolutionary algorithms via custom,
from-scratch implementations — tested against purpose-built games/simulations, with real-time
visualization to see what's happening internally and make debugging easier. See
[README.md](README.md) for the pitch and [docs/design/](docs/design/) for the numbered design docs
behind the current architecture (0001: GP engine, 0002: telemetry/visualization, 0003: algorithm
landscape and roadmap, 0004: small transformer/LM from scratch, 0005: frontend/API contracts) —
read the relevant one before an architectural change that might conflict with a decision already
made.

## Layout

- `notebooks/` — exploration, experiments, write-ups
- `apps/` — front-end apps (visualization of runs/simulations)
- `apis/` — backend APIs serving runs/simulations to `apps/`
- `libs/` — shared libraries
  - `RedQueenCbind/` — C++/pybind11 linear GP engine
  - `autodiff/` — reverse-mode automatic differentiation, built from scratch (docs/design/0004).
    The first `libs/` package another package depends on (`tinylm` depends on it) — every other
    package is a leaf.
  - `evolve/` — pure-Python evolution loop prototype (genome, fitness, selection, variation);
    zero dependency on `telemetry` (see docs/design/0001 §"Decoupling from telemetry")
  - `games/` — toy games/simulations, one module per game (e.g. `games.reach1d`), all implementing
    `evolve`'s `Environment` interface without depending on `evolve`. One package for every game
    rather than one `libs/` package per game, so shared utilities have an obvious home.
  - `telemetry/` — run registry, metrics stream, artifact store (`Protocol`-based, swappable
    backends — see docs/design/0002)
  - `tinylm/` — a small transformer LM, built on `autodiff` (docs/design/0004). Trained by
    gradients, not evolution — the odd one out relative to every other `libs/` package so far, and
    deliberately not wired into `evolve`/`telemetry` yet (see the doc for why).
- `jobs/` — training runs/workers; owns wiring a specific algorithm to `telemetry` (algorithm libs
  never import `telemetry` directly). `baseline_gp_run.py` is the reference example — run with
  `uv run python jobs/<script>.py` from the repo root.
- `docs/` — `design/000N-*.md` (numbered, one per major decision) and
  [CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) (standards + accumulated lessons)

Each directory has its own README with specifics — this file is the map, not the detail.

## Working in this repo

- **Read [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) before writing code, not after.**
- **Python tooling is `uv`**, as a workspace (root `pyproject.toml`, `[tool.uv.workspace]`, one
  shared `.venv`/`uv.lock` for every `libs/*` package). `uv sync --all-packages` installs
  everything into that one venv — plain `uv sync` only installs the (virtual, package-less) root
  project, since nothing declares the members as dependencies. Add `--extra examples` to also pull
  in `RedQueenCbind`'s example dependencies (numpy/scikit-learn). Workspace `members` is currently
  just `["libs/*"]` — new `libs/` packages join automatically; `apis/`/`jobs/` don't yet (nothing
  there needs its own `pyproject.toml` so far — `jobs/` scripts just import already-installed
  workspace packages) — add `"apis/*"`/`"jobs/*"` to `members` if/when one of them needs its own
  installable package.
- Building `RedQueenCbind` on Windows needs an MSVC dev environment (no `cl.exe` on PATH by
  default) — run `uv sync`/`uv run` through `vcvarsall.bat x64`; scikit-build-core handles the
  actual CMake/pybind11 build once the compiler is on PATH.
- Run tests/scripts via `uv run` (e.g. `uv run pytest libs/telemetry/tests`) or the venv's
  interpreter directly (`.venv/Scripts/python.exe` on Windows) — the workspace `.venv` has no
  `pip` bootstrapped into it; use `uv pip install <pkg>` for one-off additions, but prefer adding
  real dependencies to the relevant `pyproject.toml` so `uv sync` stays reproducible.
- `notebooks/` needs `uv sync --all-packages --group notebooks` (jupyter + matplotlib — not
  installed by default, so plain `uv sync` won't have them). Re-execute a notebook in place with
  `uv run jupyter execute --inplace notebooks/<name>.ipynb` so it ships with real baked-in output,
  not empty cells. On this Windows setup, `jupyter execute` reads the notebook file using the
  system locale (cp1252), not UTF-8 — it'll crash with `UnicodeDecodeError` on notebooks containing
  non-ASCII characters (e.g. em dashes) even though the file itself is valid UTF-8. Set
  `PYTHONUTF8=1` (e.g. `PYTHONUTF8=1 uv run jupyter execute --inplace ...`) to fix it.

## Keeping this file and CODING_GUIDELINES.md useful

When you hit a non-obvious mistake, gotcha, or corrected assumption that would trip up the next
agent too: add a terse (1-2 line) entry to the right doc — structural/workflow lessons here,
coding-pattern lessons in CODING_GUIDELINES.md. Bar for inclusion: **non-obvious from reading the
code**, and **general enough to recur** — not a one-off typo, and not something tests already make
impossible to repeat. Prefer folding into an existing section over appending to a growing list;
prune/consolidate rather than let either file bloat. If it's not worth a future agent's attention,
it's not worth the tokens.
