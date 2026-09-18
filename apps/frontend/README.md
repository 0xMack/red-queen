# frontend

Nuxt 4 app visualizing `apis/backend`'s runs/metrics, plus a client-side-simulated game. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and incremental plan this implements (steps 3-5).

## Contents

- `app/pages/index.vue` — run list (`GET /runs`), links to each run's detail page.
- `app/pages/runs/[id].vue` — one run's live metrics: backfills history
  (`GET /runs/{id}/metrics/history`), then opens an `EventSource` against
  `GET /runs/{id}/metrics/stream` for live updates, tearing the connection down on unmount.
- `app/pages/play/[game].vue` — play a game (`/play/snake` today) entirely client-side via Pyodide,
  running in a Web Worker (doc 0005 step 5, both phases done). This thread only translates absolute
  arrow-key presses into Snake's relative action space using a client-tracked heading (doc 0005's
  worked example) and redraws from the worker's state messages; it never touches Pyodide directly.
  Replaces the step-4 version of this page, which drove `apis/backend/routers/games.py` per tick
  instead; that router and its tests are unchanged and still valid, just no longer this page's data
  source.
- `app/workers/snakeGame.worker.ts` — the actual simulation. Loads Pyodide (CDN, pinned to match
  the installed `pyodide` npm package version exactly) and `libs/games`' real source (fetched from
  `server/api/py-games.get.ts`, written into Pyodide's virtual filesystem as the `games` package --
  so `import games.snake` runs the exact code `libs/games/tests` exercises, never a checked-in copy
  that can drift), then owns the `games.snake.Snake` instance and its `setInterval` tick loop
  off-thread. Also defines `render_state_for_js`, a small Python-side adapter (Pyodide-bridge glue,
  not part of `libs/games`) that flattens `render_state()`'s tuple-keyed `cells` dict before it
  crosses into JS -- `toJs()` raises `ConversionError` on a Python tuple used as a dict key directly
  (a real bug hit and fixed here; see `docs/CODING_GUIDELINES.md`). Talks to the main thread only
  via `postMessage` with plain, structured-cloneable objects (never a `PyProxy`, which isn't
  cloneable and is tied to this worker's own Pyodide instance).
  Superseded `app/composables/usePyodideGames.ts` (main-thread Pyodide loading), now removed --
  once the Worker version existed, keeping both would have meant one of them was dead code.
- `server/api/py-games.get.ts` — a Nitro server route reading `libs/games/src/games/*.py` straight
  off disk and returning it as JSON. Verified working both under `nuxt dev` and a built
  `.output/server/index.mjs`.
- `app/components/GridBoard.vue` — renders a `RenderState` (`{width, height, cells, score, alive}`)
  as an SVG grid. Generic over any grid game's cell labels, not Snake-specific; reused unchanged
  across every version of the play page so far (step 4's REST version, and both phases of step 5).
- `app/components/FitnessChart.vue` — a hand-rolled SVG polyline chart (best/mean fitness vs.
  generation). No charting library dependency for this skeleton -- doc 0005 named
  Nuxt/Vue/Pinia/Tailwind specifically; a charting library is a separate decision to make later if
  the hand-rolled version stops being enough.
- `app/stores/runs.ts`, `app/stores/metricsStream.ts` — Pinia stores (auto-imported by
  `@pinia/nuxt` from `app/stores/`) holding the run list and the live-metrics subscription,
  respectively.
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
`apis/backend`'s default data source).

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
  zero requests to `apis/backend` occur during play. No automated frontend test suite yet -- that's
  a gap to close before this grows much further, not a decision to leave open indefinitely.
- Pyodide's first load per browser session (inside the worker now) downloads several MB (WASM
  runtime + Python stdlib) from the CDN -- `/play/[game].vue` shows "Loading Python runtime..."
  during this. A worker is created fresh per page visit (unlike the main-thread version's
  module-level singleton) and terminated on unmount, so this download currently repeats on every
  visit to `/play/snake` -- a real tradeoff of moving to a worker, not something to silently fix by
  reintroducing a shared instance without thinking about the worker-reuse-across-navigations
  lifecycle question first.
- Worker TypeScript needs `/// <reference lib="webworker" />` plus `declare const self:
  DedicatedWorkerGlobalScope` in `snakeGame.worker.ts` -- the app's own `tsconfig` targets the DOM
  lib (for `window`/etc. elsewhere), which conflicts with the `webworker` lib if set globally; the
  triple-slash reference pulls in worker-scope types for just this one file instead.
