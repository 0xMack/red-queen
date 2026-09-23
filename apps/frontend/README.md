# frontend

Nuxt 4 app visualizing `apis/backend`'s runs/metrics, plus client-side-simulated games and trained
policies. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and incremental plan this implements (steps 3-6).

## Site map

- `/` — landing page: hero/pitch + stats, then a **watch-mode** demo: the best (or a currently
  training) Snake run's champion playing via `WatchChampion`, next to an "about this champion"
  panel (network, selection, population, fitness curve) -- visitors watch an evolved policy first
  rather than being asked to play. Then games, recent runs, and Learn chapters.
- `/games` — hub of playable games (`app/data/games.ts` drives `GameCard`s with cover images and
  facts).
- `/games/{game}` — **one page layout for every game** (docs/design/0007). `pages/games/[game].vue` only
  looks the game up; `GamePage.vue` renders the shared skeleton -- header and Watch/Play toggle, a live
  stage (with `EntrantHeader`: name, rank, representation, score ± 95% interval), a ranked side
  `LeaderboardList` ("You" slotted in by score), then the full `LeaderboardTable`, `LeaderboardPareto`
  (score vs. cost), `LeaderboardMatrix` (head to head, versus games) and `RepresentationCards`. All of it
  reads `EvaluationRecord`s (`composables/useGameBoard.ts`), never training fitness. What differs per game
  is a **`GameModule`** (`app/games/types.ts` documents the contract; `games/snake.ts`, `games/checkers.ts`,
  registered in `games/registry.ts`): the score's name and format (`ScoreSpec`), extra leaderboard columns,
  which entrant opens by default, the Watch/Play stage components, optional device probing (Snake's ONNX
  model availability), and page copy. **To add a game:** write its module and stage(s), register it, add it
  to `data/games.ts`, and produce leaderboard records with an evaluation job (`jobs/evaluate.py` for a
  single-agent score, `jobs/evaluate_versus.py` for a round robin). Stage components receive `mode`,
  `entry`, `entries`, `device` and may emit `score`/`exit`; spell those props out inline -- the SFC compiler
  can't resolve an imported type as the *whole* props type without TypeScript installed.
  - **Snake** wraps its existing pieces (`SnakeWatchStage` -> `WatchChampion`, `SnakePlayStage` ->
    `HumanPlay`) and supplies the model-package device probe.
  - **Checkers** is a two-player game, so its stage (`CheckersStage`, both Watch and Play) is built on the
    game-agnostic **versus stack** (docs/design/0006): `types/versus.ts` (`VersusEngine` = one game's
    rules, `VersusStrategy`/`Bot`), `composables/useVersusSession.ts` (seats, human click-to-move built
    from the engine's own legal moves, bot turns, history, arena, replay, and the diagnostics record) and
    `VersusStage` + `VersusSeatPicker`/`VersusMoveLog`/`VersusArena`/`VersusStrategyList` (the whole
    stage around a `#board` slot and a `#stats` slot). Layout is the same for every versus game: players, board
    and the playback controls *underneath* it on the left; seat pickers, this session's record, the game's own
    live stats (`#stats`) and the move history on the right; diagnostics, arena below. Diagnostics are also
    game-agnostic (`VersusDiagnostics`, rendered by the stage itself): one `VersusPlayerPanel` per player, side by
    side and fixed-height, each holding `VersusCandidates` (every legal move with the score its strategy gave
    it, the chosen one marked) and -- if the bot evaluates with a network -- `VersusNetwork`, the same
    `NetworkDiagram` Snake's champion uses, lit by the activations of the position the bot chose to leave
    behind (from the Rust core's own `Strategy::activations`, not a JS forward pass); below them
    `VersusEvalTrace` plots how each bot valued its chosen move, ply by ply. A bot opts in by exposing
    `scores()`, `network` and `activations()` (`types/versus.ts`). The move log is a collapsed disclosure:
    a record, not something to watch. A game supplies an engine, its strategies and a board: Checkers' are
    `utils/checkersEngine.ts` (the WASM `CheckersGame` and Rust `CheckersStrategy` players -- nothing
    re-implemented in JS) and `CheckersBoard` (deliberately not a `GridBoard` variant: pieces are captured
    and crowned rather than glide, and a human builds a multi-jump one landing at a time). Pieces have
    stable ids (`CheckersEngine` replays the core's insertion-order board semantics on them), so
    `composables/usePieceMotion.ts` -- game-agnostic -- can diff each position and replay the move as
    motion: the mover hops along its path square by square, a jumped piece shrinks away with a burst as it
    passes over, a crowned piece flashes, a new game pops the pieces in; anything it can't explain as one
    move, or a move arriving before the last finished, snaps instead of lagging. `CheckersPiece` owns a
    piece's look and CSS transitions (all off under `prefers-reduced-motion`); `CheckersBoard` adds
    coordinates, the last-move trail, capture-aware target markers, a glow in the side to move's colour and
    a result banner. Its players are
    the leaderboard's entrants (`useCheckersEntrants`: baselines are Rust players named in their id, a
    champion is fetched as plain numbers from `/runs/{id}/artifacts/{ref}/brain` -- layered weights, or a
    NEAT genome compiled to its evaluation plan -- and searched to its run's depth), so seats, arena and leaderboard speak about the
    same things. Watching opens the top entrant against the next best; playing puts you (Red) against the
    entrant you pick, and clicking another entrant swaps your opponent. **`CheckersWatch`** is the run
    viewer for a checkers run (on `/watch/{runId}` with a generation slider, and in the run page's
    champion column, pinned by its chart/table).
- **Shared across games:** `PlaybackControls` (pause / speed / new game -- speed is a *multiplier*, `utils/
  playback.ts`, over each game's base timing: Snake's tick, a versus game's pause between moves) sits under every
  game's board, in Snake's watch panel and in `VersusStage`; `BoardResult` is the one game-over overlay (Checkers'
  "Red wins" / "Draw", Snake's "Game over · N 🍎"), shown only once the board has stopped moving.
- `/learn` — an interactive textbook. `app/pages/learn.vue` is the parent route: the index renders
  full-width (search, part filter, `ChapterCard` grid with covers); a chapter gets a sidebar (chapter
  nav grouped by part + `LearnSearch`), a header generated from `app/data/learnChapters.ts`
  (number, cover, tags, prerequisites), prev/next links, a reading-progress bar, and an "on this
  page" outline built at runtime from the chapter's `<h2>`s (ids = `slugify(text)`, the same
  anchors the search's `sections` index links to). Chapter pages themselves are just the body --
  one root `<article class="prose-chapter">` (page transitions need a single root; a template
  comment next to it counts as a second one in dev). All eight chapters are written, each citing
  real code and real results (via `Callout`) rather than invented examples -- numbers come from the
  notebooks, the telemetry store, or a committed script (`jobs/checkers_round_robin.py` for the
  multi-agent chapter). Interactive pieces: `LiveSnakeDemo` (Snake case study), `ChampionProgram`
  (a real linear-GP champion, genome representations), `AutodiffPlayground` (live forward/backward
  pass on one neuron with a finite-difference check), `CausalMaskDemo` (transformers), and
  `LiveEventLog` (the real SSE stream, real-time architecture).
- `/runs` — the run list (`GET /runs`) as a sortable/filterable table: status (with a "stalled?"
  flag for `running` runs untouched for 15+ min), selection/variation, population, genome shape,
  generations vs. target, best fitness, a best-fitness sparkline, start time, duration. Sparklines
  need each run's history, fetched client-side per run (`runsStore.fetchHistories()`) -- there's no
  aggregate endpoint. Run config is interpreted in one place, `app/utils/runMeta.ts`.
- `app/pages/runs/[id].vue` — one run's live metrics: backfills history
  (`GET /runs/{id}/metrics/history`), then opens an `EventSource` against
  `GET /runs/{id}/metrics/stream` for live updates, tearing the connection down on unmount. Stat
  tiles, fitness chart (best/mean + worst..best band) and diversity chart (`LineChart.vue`), config
  + summary, a recent-generations table, and pause/resume/step controls for live runs. When the job
  recorded `held_out_score`s, a "training fitness vs. real game score" chart plots them against
  best training fitness and the leaderboard's greedy baseline -- where the lines diverge, the
  population is memorizing its training games. The
  **champion runs on the same page**: Snake runs embed `WatchChampion` (with
  `manageStream: false` -- the page already owns the stream); clicking the chart or a table row pins
  that generation's champion. Linear-GP runs show the champion program instead (`ChampionProgram`,
  parsing the artifact's `repr()` with introns marked -- `app/utils/linearProgram.ts`).
- `app/pages/games/[game].vue` — the game's one page (docs/design/0007). **Watching algorithms
  comes first**: the stage plays one leaderboard entrant (`WatchChampion`, keyed by entrant so a
  switch remounts it on the warm worker) -- by default the best *trained* model -- and the side
  `LeaderboardList` is also the selector (click any entrant; the full `LeaderboardTable` and the
  `LeaderboardPareto` chart below select too). Baselines are watchable: the worker runs
  `games.baselines` (the same code `jobs/evaluate.py` scores) via `useBaselineSession`.
  **"Play it yourself"** swaps the stage to `HumanPlay`: same board and rules, a countdown that
  waits for the runtime (`warmup` worker message), window-scoped arrow keys (or an on-screen pad on
  touch screens), a live `HumanRace` track of every entrant's mean, and `HumanResults` at the end --
  score count-up, the rank you'd take, and per-entrant "this beats it in X% of its games" bars
  (from the per-game `quality.scores` the evaluation stores). Meanwhile your row animates up the
  side list (TransitionGroup FLIP). Personal history is this browser's localStorage only
  (`utils/leaderboard.ts`); submitting humans to the real leaderboard is doc 0007 step 4. State is in
  the URL: `?watch=<entrant_id>`, `?mode=play`. Data: `GET /games/{game}/leaderboard` (records
  written by `jobs/evaluate.py`, newest protocol shown -- versions aren't comparable) and
  `GET /games/{game}/interfaces`. The run detail page links its own standing into this page, since
  "best fitness" there is training fitness, not a game score.
- `app/pages/play/[game].vue` — now just a redirect to `/games/[game]?mode=play` (old links). The
  human-steering mechanics it used to own -- absolute arrow keys translated to Snake's relative
  action space via a client-tracked heading (doc 0005's worked example) -- live in
  `useSnakeControls.ts`, shared by `HumanPlay` and `LiveSnakeDemo`.
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

### Visual design

Dark "lab" theme defined as semantic tokens in `app/assets/css/main.css` (`@theme`: `bg`/`surface`/
`raised`/`line`/`fg`/`fg-muted`, accent `queen-*`, series colors `life`/`signal`/`gold`) plus a few
`@utility` classes (`card`, `btn-primary`, `btn-ghost`, `chip`, `eyebrow`, `link`, `num`) -- use
those rather than raw Tailwind palette colors. Fonts (Space Grotesk / Inter / JetBrains Mono) load
from Google Fonts in `nuxt.config.ts`. The header is `sticky`; pages use `max-w-[1600px]` and
responsive grids, and embeddable widgets (`WatchChampion`) use container queries so they adapt to
the column they're placed in, not the viewport.

Game/chapter cover images in `public/screenshots/` are **real captures** of the running app,
regenerated with `scripts/capture_screenshots.py` (see its docstring); chapter covers without a
screenshot use `ChapterArt.vue`'s diagrams of the chapter's actual mechanism, and the Checkers card
renders a real position generated by `games.checkers` (`app/data/checkersSnapshot.ts`).

### Reusable components and session composables

Session logic (loading/error/`renderState`/`reward`/`done`/`stepCount`, and the worker-message
parsing behind them) used to be duplicated between the play and watch pages. It's now layered into
composables so every place that embeds a live Snake game -- the dedicated `/play` and `/watch`
pages, plus the landing page and the "Teaching a Snake" Learn chapter -- shares one implementation:

- `app/composables/useSnakeSession.ts` — the base: attaches to the shared worker singleton, parses
  its messages, exposes `start(spec)`/`warmup()`/`sendInput()`/`restart()`/`stop()`, plus
  per-session episode stats. `spec` is a trained network (`policyJson`), a baseline name, or
  nothing (a human plays) -- each with its interface id.
- `app/composables/useWatchSession.ts` — wraps it with the `metricsStream`-driven champion-reload
  logic described above; `useBaselineSession.ts` is the same shape for a `games.baselines` entrant.
- `app/components/HumanPlay.vue` — human play on the game page: **window-scoped** keys (it owns the
  stage while mounted; arrows/space are preventDefault'ed) via `useSnakeControls.ts`'s
  `createHeadingTracker()`, plus an on-screen pad for touch screens.
- `app/components/LiveSnakeDemo.vue` — an embeddable play-only widget (the Snake Learn chapter, and
  the landing page's fallback when no trained run is reachable) that calls `useSnakeSession()` +
  `createHeadingTracker()` directly: it needs **element-scoped** keydown
  capture (`tabindex="0"` + `event.preventDefault()` only on recognized keys) so it doesn't hijack
  page scroll/arrow keys. Starts on first click, not on mount (it's usually below the fold).
- `app/components/WatchChampion.vue` — the embeddable watch-mode counterpart, on
  `useWatchSession(runId, { manageStream })`: board, playback speed (the worker's `set_speed`
  message), per-episode score stats, and `NetworkDiagram` -- the champion's actual weights with
  live activations. The worker posts the policy's current `observation` with each watch-mode state,
  and `app/utils/snakePolicy.ts` mirrors `evolve.neuro._forward` in JS purely to *visualize* it (the
  Python copy still decides every move). `pin(generation)` loads any generation's stored champion.
  Only one Snake widget can be live per page -- they share the single worker. The champion runs
  under its run's `config.interface` (docs/design/0007): the worker builds the game with that
  interface's observer and decodes outputs with its action adapter via `games.interfaces`, so e.g.
  a 100-input grid champion sees the grid, not Snake's default features.

The two neuroevolution chapters (`/learn/neuroevolution`, `/learn/neat`) run their demos on real
algorithms in the browser, instantly: `app/utils/neuro.ts` (Evolution Strategies on a flat
weight vector) and `app/utils/neat.ts` (a TypeScript port of `libs/evolve/src/evolve/neat.py` —
genome, mutations, innovation-aligned crossover, compatibility distance, speciation, the full
generation loop; checked against the Python on identical genomes to ~1e-16, but kept in sync by
hand). Demo components: `WeightVectorExplorer`, `MutationMicroscope`, `NeuroEvoLab`,
`PermutationDemo` (competing conventions), `NeatGenomeExplorer`, `NeatCrossoverDemo`,
`SpeciationDemo`, `NeatLab` (live XOR, with a speciation on/off batch experiment), and
`NeatSnakeChampion` (a real trained NEAT champion playing Snake). `NeatDiagram.vue` draws any NEAT
genome as a graph (hidden nodes in columns by depth, disabled genes dashed) and is also what
`WatchChampion.vue` shows for a NEAT run's champion — `useWatchSession`'s `LoadedPolicy` is either
`{weights, layerSizes}` or `{genome}`; the worker loads either via `evolve.network_from_json`.
Numbers derived from `Math.tanh` and written into SVG/style attributes are rounded on purpose: Node
and the browser can disagree in the last digit, which is a hydration mismatch on a server-rendered
chapter.

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

- `app/workers/snakeGame.worker.ts` — one Snake session, entirely client-side (docs/design/0009):
  - **play**: a human steers via `postMessage`d relative actions.
  - **watch**: a model package (published champion, or one exported on demand) or a baseline
    decides every action. A finished episode auto-replays on a fresh seed after a short pause.

  The game is the Rust game core compiled to WebAssembly (`app/wasm/games/`, 47 KB, generated by
  `libs/games/build-wasm.py` and checked in so running the app needs no Rust toolchain) -- the same
  rules, observers and baselines training runs through PyO3, so a watched game is exactly the game the
  leaderboard scored. The model runs in ONNX Runtime Web in the same worker (`app/inference/runtime.ts`),
  so a decision is observation → ORT → `decodeRelative3` → `step` with no `postMessage`. Talks to the
  main thread only with plain, structured-cloneable objects -- a reactive Vue proxy (e.g. a manifest
  held in a `ref`) can't be posted, so pages send a JSON copy.

  A single instance is reused across page navigations (`app/composables/useSnakeWorker.ts`): a model
  loaded on one page is instant on the next. A page sends `{type: "stop"}` on unmount rather than
  terminating it.
- `app/workers/lm.worker.ts` + `app/inference/lm.ts` — text generation with a causal-LM package: the
  prompt in one call, then one call per character with the KV cache fed back (kept on the GPU under
  WebGPU); past the model's context window every step recomputes the window, as `tinylm.generate` does.
  Used by `TinyLMPlayground.vue` (Transformers chapter; `/dev/inference` for the scale test).
- `app/inference/` — the model runtime (docs/design/0009): `device.ts` probes WebGPU (adapter limits,
  features), WASM SIMD/threads, storage and memory (`?device=nowebgpu,nowasm,nothreads` switches
  capabilities off for testing); `match.ts` picks a package variant/backend or returns plain-language
  reasons, remembers runtime failures per device (localStorage), and asks for a click above 50 MB;
  `blobs.ts` fetches content-addressed blobs, verifies their sha256, and keeps them in the Cache API
  forever; `runtime.ts` lazily loads only the ORT build the backend needs (WASM ≈3.7 MB gzipped,
  WebGPU ≈6.6 MB) and self-tests every model against its package's parity fixture before it plays.
- `app/composables/useModelCatalog.ts` — a game's published models plus this device's match for each;
  the game page marks entrants it can't run (with the reason) and defaults to one it can.
- `app/composables/useSnakeWorker.ts` — the module-level worker singleton described above.
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
- `app/components/LineChart.vue` / `FitnessChart.vue` / `Sparkline.vue` — hand-rolled SVG charts
  (axes with nice ticks, optional band, hover tooltip, click-to-select, a ResizeObserver for crisp
  text at any width). Still no charting library dependency -- doc 0005 named
  Nuxt/Vue/Pinia/Tailwind specifically; a charting library is a separate decision to make later if
  the hand-rolled version stops being enough.
- `app/stores/runs.ts`, `app/stores/metricsStream.ts` — Pinia stores (auto-imported by
  `@pinia/nuxt` from `app/stores/`) holding the run list and the live-metrics subscription,
  respectively. `runs` also holds every run's `RunSummary` (`GET /runs/summaries`) in a `shallowRef`, replaced in
  one assignment; `ensureLoaded()` skips the request when the list is under a minute old, and `/runs` polls only
  while a run is training. The watch page reuses `metricsStream` directly, not a separate store.
- `app/types/api.gen.ts` — **generated** from `app/types/openapi.json`, the backend's OpenAPI schema
  (`uv run python apis/backend/scripts/export_openapi.py`, then `pnpm gen:api-types`;
  `apis/backend/tests/test_openapi_snapshot.py` fails while the snapshot is stale). `telemetry.ts`'s
  `RunInfo`/`GenerationStats` and `leaderboard.ts`'s `EvaluationRecord` are derived from it; what the
  backend returns as a free-form dict (`EvaluationRecord.metrics`, `InterfaceInfo`) and
  `app/types/modelpack.ts` (`libs/modelpack`'s manifest and catalog) are still kept in sync by hand.
  `games.ts`'s `RenderState`/`GridCell` are what the worker builds from the WASM core's cells.
- `app/composables/useApi.ts` — `useApi()`: `fetch(path)` and `url(path)` against the backend
  (`runtimeConfig.public.apiBase`). Every backend request goes through it.
- `pnpm typecheck` (`nuxi typecheck`, vue-tsc) — the whole app type-checks clean; keep it that way.
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
- The first model a browser runs also downloads its ONNX Runtime build (WASM ≈3.7 MB, WebGPU ≈6.6 MB
  gzipped); after that, runtimes and model blobs come from the HTTP/Cache API caches. The game core
  itself is 47 KB, so play mode starts instantly. Errors show a "Retry"; a model that failed on this
  device is remembered and explained instead of retried on every visit.
- Worker TypeScript needs `/// <reference lib="webworker" />` plus `declare const self:
  DedicatedWorkerGlobalScope` in `snakeGame.worker.ts` -- the app's own `tsconfig` targets the DOM
  lib (for `window`/etc. elsewhere), which conflicts with the `webworker` lib if set globally; the
  triple-slash reference pulls in worker-scope types for just this one file instead.
- Landing/`/games`/`/learn` verified the same way: `pnpm build` clean after each addition, then a
  live Playwright pass confirming the embedded `LiveSnakeDemo` on both the landing page and the
  Snake Learn chapter is actually playable (click to focus, arrow keys move the snake, `stepCount`
  advances), zero console errors, and `/play`/`/watch` behave identically to before being refactored
  onto `useSnakeSession`/`useWatchSession`. Installing `shiki` while a long-running
  `pnpm dev` server is still up can leave Vite's pre-bundled-deps cache stale (`504 (Outdated
  Optimize Dep)` / "Failed to fetch dynamically imported module" for the new import) -- kill the dev
  server, delete `node_modules/.cache`, restart.
