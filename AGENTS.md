# AGENTS.md

Monorepo for exploring reinforcement learning and genetic/evolutionary algorithms via custom,
from-scratch implementations — tested against purpose-built games/simulations, with real-time
visualization to see what's happening internally and make debugging easier. See
[README.md](README.md) for the pitch and [docs/design/](docs/design/) for the numbered design docs
behind the current architecture (0001: GP engine, 0002: telemetry/visualization, 0003: algorithm
landscape and roadmap, 0004: small transformer/LM from scratch, 0005: frontend/API contracts,
0006: multi-agent games and the strategy/match framework, 0007: game representations, leaderboards,
and measuring cost/tradeoffs, 0008: NEAT and tracked algorithm comparisons, 0009: client-side inference at scale — Rust/WASM
games, ONNX Runtime Web, model packages) — read the relevant one before an
architectural change that might conflict with a decision already made.

## Layout

- `notebooks/` — exploration, experiments, write-ups
- `apps/` — front-end apps (visualization of runs/simulations)
  - `frontend/` — the one Nuxt 4 app (docs/design/0005: one app, not one per concern). Run list +
    live run-detail (backfill via REST, then an `EventSource` against `apis/backend`'s SSE route)
    with a hand-rolled SVG chart. Two ways to watch a game, both via `app/workers/snakeGame.worker.ts`
    (Pyodide, in a Web Worker, off the main thread) loading real source from `libs/games`/`libs/evolve`
    (served fresh from disk by `server/api/py-source/[pkg].get.ts` — generalized from a single-package
    route once a second package needed it) rather than a checked-in copy:
    - `/games/{game}` (`snake` today) — the game's one page (docs/design/0007): watch leaderboard
      entrants play first (the leaderboard picks what's on the board — trained champions *and*
      `games.baselines`, both run in Pyodide), then "Play it yourself" on the same stage (a human
      steers; zero backend round trips per tick) with a live race against every entrant and a
      you-vs-algorithms results breakdown; the full leaderboard sits below. `/play/{game}`
      redirects here.
    - `/watch/{runId}` (docs/design/0005 step 6) — a trained `evolve.neuro.WeightVector` policy
      steers instead, loaded from the run's `champion_ref` artifact via the *existing*
      `GET /runs/{id}/artifacts/{ref}` (no backend changes needed — the endpoint is already opaque
      to what's inside). Interaction modes 2 and 3 turned out to be one mechanism (reuse
      `useMetricsStreamStore`, reload whichever champion is latest whenever it changes) rather than
      two — verified live against a real still-training run (the watched champion advanced
      gen2 → gen9 in real time).

    Both verified running in a real Worker via Playwright's `page.on("worker")`. The worker itself
    is a module-level singleton (`app/composables/useSnakeWorker.ts`), not recreated per page visit
    — Pyodide's load is memoized behind `ensurePyodideReady()` so navigating between `/play` and
    `/watch` (or revisiting either) after the first load is ~10ms instead of the ~11s cold load.
    `GridBoard.vue` renders a checkerboard board, a smoothly gliding snake (segments keyed by array
    index, not position, so a CSS `transform` transition can interpolate movement), and a pulsing
    food marker. Pinia stores live in `app/stores/` (auto-imported by `@pinia/nuxt`); `app/types/*.ts`
    mirror `apis/backend`'s response models and must be kept in sync by hand if those change. No
    automated test suite yet — verified so far with a live backend + ad hoc headless-browser
    (Playwright) checks, not checked-in tests.

    Beyond runs/games, `/` (landing), `/games`, and `/learn` (a hand-authored, foundations-first
    "interactive textbook" covering how the project's techniques actually work — code snippets,
    embedded live games, callouts citing real results) round out the site; `/runs` is where the run
    list moved. Reusable components (`CodeBlock`, `Callout`, `ChapterCard`, `GameCard`, etc.) and
    layered session composables (`useSnakeSession` → `usePlaySession`/`useWatchSession`) are a
    deliberate design principle here, not incidental — see `apps/frontend/README.md` for the full
    breakdown and why `@nuxt/content` was rejected (native-binding risk) in favor of this.
- `apis/` — backend APIs serving runs/simulations to `apps/`
  - `backend/` — the one FastAPI service (docs/design/0005: one module, not one per concern, until
    something forces a split). `routers/runs.py` wraps `telemetry` directly, reusing its pydantic
    models (`RunInfo`, `GenerationStats`) as response models rather than duplicating schemas. The
    `/metrics/stream` SSE route adapts `telemetry.FileMetricsStore.subscribe()` — a synchronous,
    never-returning polling generator — to async via `anyio.to_thread.run_sync(...,
    abandon_on_cancel=True)`; see `apis/backend/README.md` for why that route's happy path is
    tested at the generator level, not through a live `TestClient` request (Starlette's TestClient
    doesn't reliably simulate a mid-stream disconnect, so a full request hangs). `routers/games.py`
    runs a game session's `Environment` server-side in an in-memory `GameSessionStore`
    (`game_sessions.py`) — no telemetry/persistence, a session doesn't survive a restart. Only
    games implementing `games.rendering.Renderable` are registered (`snake` today). `POST
    /runs/{id}/control` (pause/resume/step) reuses `telemetry.RunStatus`'s existing `"paused"`
    value as the only coordination with a training job's `jobs/control.py` callback — see
    `docs/CODING_GUIDELINES.md`'s "before adding new shared state" entry for why, and why `step`
    is implemented entirely API-side instead of as a third status value.
- `libs/` — shared libraries
  - `RedQueenCbind/` — C++/pybind11 linear GP engine
  - `autodiff/` — reverse-mode automatic differentiation, built from scratch (docs/design/0004).
    The first `libs/` package another package depends on (`tinylm` depends on it) — every other
    package is a leaf.
  - `evolve/` — pure-Python evolution loop prototype (genome, fitness, selection, variation);
    zero dependency on `telemetry` (see docs/design/0001 §"Decoupling from telemetry"). Every
    submodule is pure stdlib (verified before `apps/frontend`'s Pyodide bridge loaded the whole
    package client-side, docs/design/0005 step 6) — no numpy, no external deps.
    `WeightVector.to_json()`/`from_json()` is the real (round-trippable) wire format for a trained
    policy — `jobs/snake_neuro_run.py` writes it, the Pyodide bridge loads it back with the exact
    same code. `match.py` (docs/design/0006) is the two-player sibling of `simulation.py`'s
    single-agent `Environment`/`SimulationFitnessEvaluator`: `MultiAgentEnvironment`
    (`reset()`/`legal_moves()`/`current_player()`/`step(move)`/`winner()`), `play_match()` (the
    "pit any strategy against any strategy" mechanic, reusable across every multi-agent game, not
    checkers-specific), and `MatchFitnessEvaluator` (fitness from match outcomes against reference
    opponents). No new `Strategy` class hierarchy — just `(observation, legal_moves) -> move`, the
    same shape `act(genome, observation) -> action` already used for Snake, just wider.
  - `games/` — toy games/simulations, one module per game (e.g. `games.reach1d`), all implementing
    `evolve`'s `Environment` interface (or, for the one two-player game so far,
    `evolve.match.MultiAgentEnvironment`) without depending on `evolve`. One package for every game
    rather than one `libs/` package per game, so shared utilities have an obvious home. `snake.py`'s
    observation is 11 hand-engineered features (danger/heading/food-direction), not a raw flattened
    grid — swapped after a real trained result confirmed the earlier representation, not compute,
    was the ceiling (docs/CODING_GUIDELINES.md's "Simulations / RL environments"). `checkers.py`
    (docs/design/0006 phase 1) — real rules (mandatory captures, mandatory multi-jump chains,
    kinging), a `Move` is a whole turn so `current_player()` alternates strictly every `step()`;
    `render_state()` reuses `snake.py`'s exact `{width, height, cells}` shape, just a wider label
    vocabulary (`"black_man"`/`"red_king"`/...) — no protocol changes needed for two-colored,
    two-typed pieces. Validated: two static strategies play a full match via `play_match()`, and a
    small evolved population measurably improves against a randomized opponent via
    `MatchFitnessEvaluator` (noisy with too few opponent samples, clean with more — see
    docs/CODING_GUIDELINES.md).
  - `evolve/neat.py` (docs/design/0008) — NEAT: a graph genome of innovation-numbered connection genes
    whose *structure* evolves (add connection / split a connection into a node), with
    innovation-aligned crossover, speciation, and fitness sharing. Has its own loop
    (`evolve_neat()`), because speciation is population-level, but reports through `evolve()`'s
    `GenerationSummary` (`extras` carries species count and champion size) so every `jobs/` callback
    works unchanged. `evolve.network_from_json()` loads either network kind from an artifact.
    `apps/frontend/app/utils/neat.ts` is a hand-kept TypeScript port (for the Learn demos and for
    drawing champions) — forward passes are checked against the Python to ~1e-16.
  - `telemetry/` — run registry, metrics stream, artifact store (`Protocol`-based, swappable
    backends — see docs/design/0002), plus `EvaluationStore` (`evaluations.db`): leaderboard
    results per (protocol, entrant), docs/design/0007.
  - **Representations (docs/design/0007)**: a game's observation is chosen by an *observer*, and a
    model's outputs become moves via an *action adapter*; the pair is a versioned **interface**
    (`snake/features.v1+relative3.v1`, `snake/grid-flat.v1+relative3.v1`), registered in
    `games.interfaces` and recorded as `config.interface` on every game run. A champion only runs
    under its own interface — never edit an observer's encoding in place; add a new version (an
    in-place rewrite is what once broke every older Snake champion).
  - `tinylm/` — a small transformer LM, built on `autodiff` (docs/design/0004). Trained by
    gradients, not evolution — the odd one out relative to every other `libs/` package so far, and
    deliberately not wired into `evolve`/`telemetry` yet (see the doc for why).
- `jobs/` — training runs/workers; owns wiring a specific algorithm to `telemetry` (algorithm libs
  never import `telemetry` directly). `baseline_gp_run.py` is the reference example (linear GP);
  `snake_neuro_run.py` is the same neuroevolution-vs.-Snake setup validated in
  `notebooks/0005-neuroevolution-snake.ipynb`, wired to telemetry — its champions are what
  `apps/frontend`'s `/watch/{runId}` loads. Run with `uv run python jobs/<script>.py` from the repo
  root. `snake_neuro_run.py [interface_id] [--seeds fixed:5|resample:N]` trains under any registered
  interface with a training-seed strategy (`seeding.py`), records a measured training-cost block
  (`costs.py`) in the run summary, and logs a held-out game score every N generations (the
  overfitting curve on the run page). `evaluate.py` produces the
  leaderboards (docs/design/0007): every finished game run's champion plus fixed baselines, on
  held-out seeds under a versioned protocol, with inference/training cost — never training fitness.
  `backfill_interfaces.py` is a one-off for runs recorded before interfaces existed.
  `snake_neat_run.py` is the NEAT counterpart of `snake_neuro_run.py`; `snake_experiment.py` runs a
  tagged (arm × rng seed) comparison of the two and aggregates it (`GenerationStats.extras`,
  `config.experiment`/`arm` are how NEAT-specific curves and experiment groups are tracked).
  `control.py`'s `make_control_callback`
  (an `on_generation` entry) is the job-side half of the pause/resume control API — see the
  `apis/backend` bullet above. Not a `uv` workspace package (no `pyproject.toml`) — scripts here
  import each other as plain sibling modules, which works because `uv run python jobs/<script>.py`
  puts the script's own directory on `sys.path`; `jobs/tests/conftest.py` does the same explicitly
  so `uv run pytest jobs/tests` can too.
- `docs/` — `design/000N-*.md` (numbered, one per major decision) and
  [CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) (standards + accumulated lessons)

Each directory has its own README with specifics — this file is the map, not the detail.

## Working in this repo

- **Read [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) before writing code, not after.**
- **Python tooling is `uv`**, as a workspace (root `pyproject.toml`, `[tool.uv.workspace]`, one
  shared `.venv`/`uv.lock` for every `libs/*` package). `uv sync --all-packages` installs
  everything into that one venv — plain `uv sync` only installs the (virtual, package-less) root
  project, since nothing declares the members as dependencies. Add `--extra examples` to also pull
  in `RedQueenCbind`'s example dependencies (numpy/scikit-learn). Workspace `members` is
  `["libs/*", "apis/*"]` — new packages under either join automatically. `jobs/` still doesn't have
  its own `pyproject.toml` (scripts there just import already-installed workspace packages) — add
  `"jobs/*"` to `members` if that changes.
- **`apps/frontend` is a separate `pnpm` project**, not part of the `uv` workspace — its own
  `node_modules`/`pnpm-lock.yaml`. `pnpm install` / `pnpm dev` from `apps/frontend/`. No JS
  workspace at the repo root yet since there's only one JS package.
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
