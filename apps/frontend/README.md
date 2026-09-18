# frontend

Nuxt 4 app visualizing `apis/backend`'s runs/metrics, plus a client-side-simulated game. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and incremental plan this implements (steps 3-5).

## Contents

- `app/pages/index.vue` — run list (`GET /runs`), links to each run's detail page.
- `app/pages/runs/[id].vue` — one run's live metrics: backfills history
  (`GET /runs/{id}/metrics/history`), then opens an `EventSource` against
  `GET /runs/{id}/metrics/stream` for live updates, tearing the connection down on unmount.
- `app/pages/play/[game].vue` — play a game (`/play/snake` today) entirely client-side via Pyodide
  (doc 0005 step 5, main-thread phase): `usePyodideGames.ts` loads a real CPython-in-WASM runtime
  and the actual `games.snake.Snake` class runs in the browser, ticked by a local `setInterval` --
  no backend round trip per tick (verified: zero requests to `apis/backend` during play). This
  replaces the step-4 version of this page, which drove `apis/backend/routers/games.py` instead;
  that router and its tests are unchanged and still valid, just no longer this page's data source.
  Controls translate absolute arrow keys into Snake's relative action space using a client-tracked
  heading (doc 0005's worked example), not the raw &larr;/&rarr;-only scheme step 4 used.
- `app/composables/usePyodideGames.ts` — loads Pyodide once per browser session (module-level
  singleton) from the official CDN (`cdn.jsdelivr.net/pyodide`, pinned to match the installed
  `pyodide` npm package version exactly), fetches `libs/games`' real source from
  `server/api/py-games.get.ts`, and writes it into Pyodide's virtual filesystem as the `games`
  package -- so `import games.snake` in the browser runs the exact code `libs/games/tests`
  exercises, never a checked-in copy that can drift. Also defines `render_state_for_js`, a small
  Python-side adapter (Pyodide-bridge glue, not part of `libs/games`) that flattens
  `render_state()`'s tuple-keyed `cells` dict before it crosses into JS -- `toJs()` raises
  `ConversionError` on a Python tuple used as a dict key directly (a real bug hit and fixed here;
  see `docs/CODING_GUIDELINES.md`).
- `server/api/py-games.get.ts` — a Nitro server route reading `libs/games/src/games/*.py` straight
  off disk and returning it as JSON. Verified working both under `nuxt dev` and a built
  `.output/server/index.mjs`.
- `app/components/GridBoard.vue` — renders a `RenderState` (`{width, height, cells, score, alive}`)
  as an SVG grid. Generic over any grid game's cell labels, not Snake-specific; reused unchanged
  between the step-4 (REST) and step-5 (Pyodide) versions of the play page.
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
  the chart with zero console errors. `/play/snake` (the Pyodide version) was verified the same
  way: Python runtime loads, real keyboard input changes the snake's heading and path, a wall
  collision ends the episode and shows "Game over", "Play again" starts a fresh episode -- and,
  checked explicitly, zero requests to `apis/backend` occur during play. No automated frontend test
  suite yet -- that's a gap to close before this grows much further, not a decision to leave open
  indefinitely.
- Pyodide's first load per browser session downloads several MB (WASM runtime + Python stdlib)
  from the CDN -- the `/play/[game].vue` page shows "Loading Python runtime..." during this. Every
  page after the first reuses the already-loaded instance instantly (`usePyodideGames.ts`'s
  module-level singleton).
