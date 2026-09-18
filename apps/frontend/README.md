# frontend

Nuxt 4 app visualizing `apis/backend`'s runs/metrics, plus client-side-simulated games and trained
policies. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and incremental plan this implements (steps 3-6).

## Site map

- `/` — landing page: hero/pitch, an embedded live `LiveSnakeDemo`, links into Games/Learn/Runs.
- `/games` — hub of playable games (`app/data/games.ts` drives the card grid; Checkers shows as
  "coming soon" until doc 0006 phases 2-4 land).
- `/learn` — an interactive-textbook "wiki" covering how the project's techniques actually work,
  foundations first (`app/data/learnChapters.ts` drives the chapter index; most chapters are
  "coming soon" placeholders, filled in incrementally). Three chapters are fully written today:
  `genetic-algorithms.vue`, `selection-strategies.vue`, `teaching-a-snake.vue` — each cites real
  code from `libs/evolve`/`games/snake.py` and real notebook/training results (via `Callout`), not
  invented examples. `teaching-a-snake.vue` embeds a live, playable `LiveSnakeDemo`.
- `/runs` — the run list (`GET /runs`), links to each run's detail page. Moved here from `/` when
  `/` became the landing page above — behavior unchanged.
- `app/pages/runs/[id].vue` — one run's live metrics: backfills history
  (`GET /runs/{id}/metrics/history`), then opens an `EventSource` against
  `GET /runs/{id}/metrics/stream` for live updates, tearing the connection down on unmount. Links
  to `/watch/{id}` when `run.config.game === "snake"`.
- `app/pages/play/[game].vue` — play a game (`/play/snake` today) entirely client-side via Pyodide,
  running in a Web Worker (doc 0005 step 5). This thread only translates absolute arrow-key presses
  into Snake's relative action space using a client-tracked heading (doc 0005's worked example) and
  redraws from the worker's state messages; it never touches Pyodide directly. Replaces the step-4
  version of this page, which drove `apis/backend/routers/games.py` per tick instead; that router
  and its tests are unchanged and still valid, just no longer this page's data source. Now a thin
  page composing `usePlaySession()` (below) rather than owning worker plumbing itself.
- `app/pages/watch/[runId].vue` — watch a trained policy play, no controls (doc 0005 step 6,
  interaction modes 2 and 3 -- turn out to be one mechanism, not two: reuses
  `useMetricsStreamStore` and re-loads whichever champion is latest whenever it changes. For a
  finished run that's once; for a still-training run, every time a new `GenerationStats` arrives
  over SSE. No separate code path for "watch a specific finished run" vs. "watch training live" --
  verified both, including a real live run where the watched champion advanced gen2 → gen9 in real
  time as training progressed). Fetches the champion artifact via the *existing*
  `GET /runs/{id}/artifacts/{ref}` -- no `apis/backend` changes needed at all, since a serialized
  `WeightVector` is just opaque bytes to `ArtifactStore`, same as a `LinearProgram`'s `repr()`. Now
  a thin page composing `useWatchSession(runId)` (below).

### Reusable components and session composables

Session logic (loading/error/`renderState`/`reward`/`done`/`stepCount`, and the worker-message
parsing behind them) used to be duplicated between the play and watch pages. It's now layered into
composables so every place that embeds a live Snake game -- the dedicated `/play` and `/watch`
pages, plus the landing page and the "Teaching a Snake" Learn chapter -- shares one implementation:

- `app/composables/useSnakeSession.ts` — the base: attaches to the shared worker singleton, parses
  its messages, exposes `start()`/`sendInput()`/`restart()`/`stop()`.
- `app/composables/usePlaySession.ts` — wraps it with a **window-scoped** keydown listener (owns
  the whole viewport, e.g. a dedicated page) via `useSnakeControls.ts`'s `createHeadingTracker()`.
- `app/composables/useWatchSession.ts` — wraps it with the `metricsStream`-driven champion-reload
  logic described above.
- `app/components/LiveSnakeDemo.vue` — an embeddable play-only widget (used on the landing page and
  in the Snake Learn chapter) that calls `useSnakeSession()` + `createHeadingTracker()` directly,
  **not** `usePlaySession()`: it needs **element-scoped** keydown capture (`tabindex="0"` +
  `event.preventDefault()` only on recognized keys) so it doesn't hijack page scroll/arrow keys
  until a visitor actually clicks into it. It's deliberately play-mode only for now -- nothing
  embeds a watch-mode demo yet, and `useWatchSession` is proven and ready whenever something does.

Other components in `app/components/`, used across the games/learn pages: `GameStatRow.vue` (the
score/step/reward readout, extracted from its duplicated form in the play/watch pages),
`GameCard.vue`/`ChapterCard.vue` (index cards with a `status: "available" | "coming-soon"` prop, so
unwritten chapters/unbuilt games render dimmed and unlinked instead of being omitted),
`CodeBlock.vue` (syntax-highlighted snippets via `app/composables/useHighlighter.ts`, a `shiki`
fine-grained-bundle singleton -- explicit langs (python/typescript/bash/json) and the JS regex
engine, not the full bundle or WASM oniguruma, to keep this lean and native-binding-free), and
`Callout.vue` (`variant: "note" | "warning" | "finding"` -- `"finding"` flags a real bug/result the
prose references, e.g. Snake's reward-hacking or its representation-ceiling result).

Deliberately **not** `@nuxt/content`: it pulls in a SQLite-backed content database
(`better-sqlite3`) as a peer dependency, real native-binding risk on this Windows machine given the
already-documented MSVC/`RedQueenCbind` build friction, for what's currently ~6-10 pages. Hand-
authored Vue pages + the component library above both sidesteps that risk and makes "reusable
components" the literal mechanism rather than something hidden behind a markdown-rendering layer.
Revisit if the chapter count grows enough that hand-authoring markup becomes the bottleneck.

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

  A single instance is reused across page navigations (`app/composables/useSnakeWorker.ts`), not
  recreated per visit -- Pyodide's own loading is memoized behind `ensurePyodideReady()` so a
  second `"start"` message on an already-loaded worker skips the multi-second CDN download/WASM
  init entirely (measured: ~11s cold, ~10ms warm). A page sends `{type: "stop"}` on unmount rather
  than terminating the worker, which just pauses its tick loop (and cancels any pending watch
  auto-replay) without discarding the loaded runtime.
- `app/composables/useSnakeWorker.ts` — the module-level worker singleton described above.
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
  as an SVG grid: a checkerboard background (one `<pattern>`, not one `<rect>` per background
  cell), rounded snake segments with a distinct head (directional eyes, inferred from the head's
  position relative to the segment behind it -- purely cosmetic, never affects gameplay), a
  pulsing circular food marker, and a smooth glide between ticks. The glide needs body segments
  keyed by **array index**, not `x,y` position -- `render_state()`'s `cells` are always ordered
  [segment nearest the head, ..., tail, head] (see `games/snake.py`'s `_cell_labels()`), so index
  `i` consistently refers to the same physical body link tick to tick, letting a CSS `transform`
  transition interpolate its movement. Food is keyed separately (a stable literal key, no
  transition) so it teleports to its new cell when eaten rather than sliding there. One known,
  accepted minor artifact: the segment nearest the head can snap instead of glide for one frame
  right after eating, since growth shifts every index by one. Generic over any grid game's cell
  labels, not Snake-specific; reused unchanged across every play-mode version and the watch page.
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
  trained run (`jobs/snake_neuro_run.py`, 250 generations, `games.snake`'s improved 11-feature
  observation + `LexicaseSelection` -- best_fitness 0.65 → 17.28 over the earlier setup): the loaded
  policy actually eats food (score up to 9 observed in a single episode, not just surviving),
  auto-replays once its episode ends, and, watched against a real *still-training* run, correctly
  re-loaded a new champion live as each generation was recorded (observed 8 distinct champion refs
  advance gen2 → gen9 in real time). No automated frontend test suite yet -- that's a gap to close
  before this grows much further, not a decision to leave open indefinitely.
- Pyodide's first load per browser session downloads several MB (WASM runtime + Python stdlib) from
  the CDN (~11s measured) -- `/play/[game].vue` and `/watch/[runId].vue` both show "Loading Python
  runtime..." during this. The worker is a module-level singleton reused across page navigations
  (`app/composables/useSnakeWorker.ts`), so this cost is paid once per browser session, not once per
  visit -- a subsequent navigation to either page measured ~10ms to be ready. Both pages also have a
  "Retry" button on error (e.g. the CDN being unreachable) that re-attempts the same startup path
  rather than requiring a full page reload.
- Worker TypeScript needs `/// <reference lib="webworker" />` plus `declare const self:
  DedicatedWorkerGlobalScope` in `snakeGame.worker.ts` -- the app's own `tsconfig` targets the DOM
  lib (for `window`/etc. elsewhere), which conflicts with the `webworker` lib if set globally; the
  triple-slash reference pulls in worker-scope types for just this one file instead.
- Landing/`/games`/`/learn` verified the same way: `pnpm build` clean after each addition, then a
  live Playwright pass confirming the embedded `LiveSnakeDemo` on both the landing page and the
  Snake Learn chapter is actually playable (click to focus, arrow keys move the snake, `stepCount`
  advances), zero console errors, and `/play`/`/watch` behave identically to before being refactored
  onto `useSnakeSession`/`usePlaySession`/`useWatchSession`. Installing `shiki` while a long-running
  `pnpm dev` server is still up can leave Vite's pre-bundled-deps cache stale (`504 (Outdated
  Optimize Dep)` / "Failed to fetch dynamically imported module" for the new import) -- kill the dev
  server, delete `node_modules/.cache`, restart.
