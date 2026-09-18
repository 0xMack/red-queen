# frontend

Nuxt 4 app visualizing `apis/backend`'s runs/metrics, plus client-side-simulated games and trained
policies. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and incremental plan this implements (steps 3-6).

## Contents

- `app/pages/index.vue` — run list (`GET /runs`), links to each run's detail page.
- `app/pages/runs/[id].vue` — one run's live metrics: backfills history
  (`GET /runs/{id}/metrics/history`), then opens an `EventSource` against
  `GET /runs/{id}/metrics/stream` for live updates, tearing the connection down on unmount. Links
  to `/watch/{id}` when `run.config.game === "snake"`.
- `app/pages/play/[game].vue` — play a game (`/play/snake` today) entirely client-side via Pyodide,
  running in a Web Worker (doc 0005 step 5). This thread only translates absolute arrow-key presses
  into Snake's relative action space using a client-tracked heading (doc 0005's worked example) and
  redraws from the worker's state messages; it never touches Pyodide directly. Replaces the step-4
  version of this page, which drove `apis/backend/routers/games.py` per tick instead; that router
  and its tests are unchanged and still valid, just no longer this page's data source.
- `app/pages/watch/[runId].vue` — watch a trained policy play, no controls (doc 0005 step 6,
  interaction modes 2 and 3 -- turn out to be one mechanism, not two: reuses
  `useMetricsStreamStore` and re-loads whichever champion is latest whenever it changes. For a
  finished run that's once; for a still-training run, every time a new `GenerationStats` arrives
  over SSE. No separate code path for "watch a specific finished run" vs. "watch training live" --
  verified both, including a real live run where the watched champion advanced gen2 → gen9 in real
  time as training progressed). Fetches the champion artifact via the *existing*
  `GET /runs/{id}/artifacts/{ref}` -- no `apis/backend` changes needed at all, since a serialized
  `WeightVector` is just opaque bytes to `ArtifactStore`, same as a `LinearProgram`'s `repr()`.
- `app/workers/snakeGame.worker.ts` — the actual simulation, in one of two modes:
  - **play**: a human steers via `postMessage`d relative actions.
  - **watch**: a loaded `evolve.neuro.WeightVector` policy decides every action instead. A
    finished episode auto-replays with a fresh random seed after a short pause (so a completed
    run's watch page doesn't just freeze forever once its one champion dies) -- immediately
    superseded if a genuinely new champion arrives first.

  Either way: loads Pyodide (CDN, pinned to match the installed `pyodide` npm package version
  exactly) and the real source of both `libs/games` (`import games.snake`) and, for watch mode,
  the *whole* `libs/evolve` package (`evolve/__init__.py` eagerly imports every submodule, and all
  of them are pure stdlib -- verified before wiring this up, no numpy/micropip needed), fetched
  from `server/api/py-source/[pkg].get.ts` and written into Pyodide's virtual filesystem -- so
  Pyodide runs the exact code `libs/games/tests`/`libs/evolve/tests` exercise, never a checked-in
  copy that can drift. Also defines, as Pyodide-bridge glue (not part of `libs/games`/`libs/evolve`
  themselves): `render_state_for_js` (flattens `render_state()`'s tuple-keyed `cells` dict before it
  crosses into JS -- `toJs()` raises `ConversionError` on a Python tuple used as a dict key directly,
  a real bug hit and fixed here; see `docs/CODING_GUIDELINES.md`), `load_weight_vector` (JSON text →
  `WeightVector`), and `snake_policy_action`/`watch_tick` (argmax-3-outputs-to-turn -- must match
  `jobs/snake_neuro_run.py`'s `act()` exactly, since that's the convention every champion this loads
  was actually trained under). Talks to the main thread only via `postMessage` with plain,
  structured-cloneable objects (never a `PyProxy`, which isn't cloneable and is tied to this
  worker's own Pyodide instance).
- `server/api/py-source/[pkg].get.ts` — a Nitro server route reading a `libs/<pkg>` package's real
  source straight off disk and returning it as JSON (`pkg` is `games` or `evolve`). Generalizes
  what was a single-purpose `py-games.get.ts` (doc 0005 step 4) once a second Pyodide-loaded
  package showed up. Resolves paths from `process.cwd()`, not `import.meta.url` -- a real, verified-
  by-running discrepancy: Nitro's dev-mode bundler doesn't preserve this route's actual source-tree
  depth once it moved into a subdirectory (`py-source/[pkg].get.ts` vs. the old flat
  `py-games.get.ts`), so an `import.meta.url`-relative path landed two directories too high, not
  the one extra level the added nesting would predict. `process.cwd()` sidesteps that; both
  `nuxt dev` and a built `.output/server/index.mjs` are run from `apps/frontend`, so
  `cwd + "../../libs"` is reliable there.
- `app/components/GridBoard.vue` — renders a `RenderState` (`{width, height, cells, score, alive}`)
  as an SVG grid. Generic over any grid game's cell labels, not Snake-specific; reused unchanged
  across every play-mode version and the watch page.
- `app/components/FitnessChart.vue` — a hand-rolled SVG polyline chart (best/mean fitness vs.
  generation). No charting library dependency for this skeleton -- doc 0005 named
  Nuxt/Vue/Pinia/Tailwind specifically; a charting library is a separate decision to make later if
  the hand-rolled version stops being enough.
- `app/stores/runs.ts`, `app/stores/metricsStream.ts` — Pinia stores (auto-imported by
  `@pinia/nuxt` from `app/stores/`) holding the run list and the live-metrics subscription,
  respectively. The watch page reuses `metricsStream` directly, not a separate store.
- `app/types/telemetry.ts`, `app/types/games.ts` — TypeScript interfaces mirroring
  `apis/backend`'s response models (`telemetry.RunInfo`/`GenerationStats` reused directly;
  `backend.schemas.GameSessionState` et al.) — `games.ts`'s `RenderState`/`GridCell` also double as
  the Pyodide bridge's shape, since `render_state_for_js`'s output matches
  `game_sessions.json_safe_render_state()`'s shape exactly. Keep in sync by hand if those change.
- `nuxt.config.ts` — `runtimeConfig.public.apiBase` (default `http://127.0.0.1:8000`, override via
  `NUXT_PUBLIC_API_BASE` -- copy `.env.example` to `.env`), Tailwind CSS 4 wired in via
  `@tailwindcss/vite` (not the `@nuxtjs/tailwindcss` module), Pinia via `@pinia/nuxt`.

## Running it

Needs `apis/backend` running first (see `apis/backend/README.md`) and at least one recorded run
(`uv run python jobs/baseline_gp_run.py` from the repo root writes to `jobs/run-data`, which is
`apis/backend`'s default data source; `uv run python jobs/snake_neuro_run.py` additionally gives
you a `/watch/{id}`-able run, though it takes a few minutes).

```bash
pnpm install
pnpm dev
```

Then open `http://localhost:3000`.

## Notes

- Package manager is `pnpm` (this app's own `node_modules`/lockfile -- separate from the repo's
  Python `uv` workspace, no JS workspace at the repo root yet since there's only one JS package).
- Verified with a live backend + a headless-browser check (Playwright, run ad hoc, not checked in
  as a test yet): the run list renders real data via SSR, navigating to a run detail page opens a
  real SSE connection (`connected` flips to `live`), backfills existing generations, and renders
  the chart with zero console errors. `/play/snake` was verified the same way, including confirming
  Playwright's `page.on("worker")` event actually fires for `snakeGame.worker.ts`: Python runtime
  loads inside it, real keyboard input changes the snake's heading and path, a wall collision ends
  the episode and shows "Game over", "Play again" starts a fresh episode -- and, checked explicitly,
  zero requests to `apis/backend` occur during play. `/watch/{runId}` was verified against a real
  trained run (150 real generations, `jobs/snake_neuro_run.py`): the loaded policy actually plays
  (survives varying numbers of steps depending on the random seed -- an honest, sometimes-short
  result, not tuned to look good), auto-replays once its episode ends, and, watched against a real
  *still-training* run, correctly re-loaded a new champion live as each generation was recorded
  (observed 8 distinct champion refs advance gen2 → gen9 in real time). No automated frontend test
  suite yet -- that's a gap to close before this grows much further, not a decision to leave open
  indefinitely.
- Pyodide's first load per browser session (inside the worker now) downloads several MB (WASM
  runtime + Python stdlib) from the CDN -- `/play/[game].vue` and `/watch/[runId].vue` both show
  "Loading Python runtime..." during this. A worker is created fresh per page visit (unlike the old
  main-thread version's module-level singleton) and terminated on unmount, so this download
  currently repeats on every visit -- a real tradeoff of moving to a worker, not something to
  silently fix by reintroducing a shared instance without thinking about the
  worker-reuse-across-navigations lifecycle question first.
- Worker TypeScript needs `/// <reference lib="webworker" />` plus `declare const self:
  DedicatedWorkerGlobalScope` in `snakeGame.worker.ts` -- the app's own `tsconfig` targets the DOM
  lib (for `window`/etc. elsewhere), which conflicts with the `webworker` lib if set globally; the
  triple-slash reference pulls in worker-scope types for just this one file instead.
