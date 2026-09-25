# Coding Guidelines

Standards for this repo, plus accumulated lessons (see [AGENTS.md](../AGENTS.md) for the bar on
what belongs here and how to add to it). Read before writing code, not after.

## General

- Keep abstractions pluggable only at proven extension points (`Program`, `FitnessEvaluator`,
  `SelectionStrategy`, `VariationStrategy` per docs/design/0001) — don't invent new ones
  speculatively.
- Dependency direction: algorithm/lib packages never import `telemetry`; only integration glue
  (usually in `jobs/`) imports both. See docs/design/0002.
- Before adding new shared state for cross-process coordination, check whether an existing field
  already means what you need. `apis/backend`'s pause/resume control (docs/design/0005 step 7)
  needed the API process and a separate training-job process to agree on "should this run keep
  going right now?" — `telemetry.RunStatus` already had a `"paused"` value from docs/design/0002,
  unused until then, so pausing became `registry.update_status(run_id, "paused")` and the job's
  `on_generation` callback (`jobs/control.py`) just blocks while that's true. No new IPC primitive,
  no new column. "Step" (advance exactly one generation) resisted the same trick — it's not
  sensibly a `RunStatus` value — so it's implemented entirely API-side instead (resume, wait for one
  new `GenerationStats` via the existing `MetricsSource`, re-pause), keeping the job-side callback a
  two-state check rather than growing a third state for one caller's benefit.
- **Formatting is enforced, so just run it**: `uv run ruff format . && uv run ruff check --fix .` and `cargo fmt`
  (in `libs/games`) before committing -- CI fails otherwise. One config at 120 columns (root `pyproject.toml`,
  `libs/games/rustfmt.toml`); every package's `[tool.ruff]` must `extend` the root, because ruff uses the
  *nearest* `pyproject.toml` and doesn't inherit -- the per-package copies silently kept everything under
  `libs/`/`apis/` at the 88-column default. The one-time reformat is in `.git-blame-ignore-revs`.

## Python

- **Data models: pydantic, not bare dataclasses/TypedDicts**, for anything crossing a boundary
  (API payloads, stored records, config). Every field gets `Field(..., description=...)` and real
  validation (`ge=`/`gt=`, `min_length=`, `Literal`/`Enum`, or a `field_validator`) — not just a
  type hint. A bare type hint doesn't self-document and doesn't catch bad data at the boundary; a
  `Field` does both. Example: `libs/telemetry/src/telemetry/types.py`.
- Pluggable interfaces: `typing.Protocol`, not ABCs — matches the structural style already used in
  `libs/telemetry`.
- Tests: pytest, `tmp_path` fixture for anything touching the filesystem — no shared mutable state
  between tests.

## C++

- Never construct a `random_device`/`mt19937` (or any RNG) per call — reuse a `static` engine.
  Real perf bug we hit: it was happening 4x per instruction, 16 instructions per individual.
- A constructor parameter shadowing a member (`mutationRate = mutationRate;`) silently discards
  the argument — use `this->member = param` or distinct names. The compiler won't warn you.
- Don't `new` a value type just to `push_back` a copy into a container — construct in place
  (`emplace_back()`). The `new` here was an unfreed leak, one per program instruction.
- If a buffer is sized once and then only ever appended to via `back_inserter`/`push_back`, the
  initial size is dead weight and misleading — either size it correctly and write in place, or
  don't preallocate.
- Any per-item state that persists across items in a loop (e.g. registers across dataset rows)
  must be explicitly reset at the start of each item unless carrying over is intended — it was
  silently carrying over here and produced wrong predictions for every row after the first.

## Simulations / RL environments

- An environment's observation must contain enough information to determine the correct action —
  two episodes that differ only in something not exposed in the observation (e.g. the target, in
  `libs/games/src/games/reach1d.py`) are indistinguishable to any policy, no matter how it's
  trained/evolved. Found by running neuroevolution against `reach1d` and noticing two symmetric
  benchmark scenarios could never both improve; fixed by making the observation relative to the
  goal instead of absolute. Check this before spending compute tuning an algorithm against a new
  environment.
- A symmetric shaping reward (`+x` for progress, `-x` for regress) can be reward-hacked into
  oscillating in place forever for ~0 net reward — a real, lower-risk local optimum than continuing
  to pursue the sparse objective. Found in `games/snake.py`: an evolved policy learned to bounce
  between two cells instead of continuing toward food. Fixed by making the penalty for regress
  larger than the reward for progress, so standing still is strictly worse than seeking the goal,
  not merely no-better. Check what a policy can gain by doing nothing before trusting a shaped
  reward is safe.
- Selecting on aggregate/mean fitness across multiple cases can make the population better on
  average while making a specific champion worse on a specific case than an earlier, weaker-on-
  average one was — not a bug, a real property of aggregate selection (the same tension
  `LexicaseSelection` exists to address, docs/design/0003). Seen scaling up `games/snake.py`
  training: mean fitness rose and a total-failure scenario got fixed, but one previously-fine
  scenario got worse. Don't assume "better on average" implies "better everywhere" when reporting
  or comparing runs.
- A raw board-flattened observation (one float per grid cell) makes a small evolved network do two
  jobs at once: learn to extract structure ("is there a wall two cells ahead?", "which way is the
  food?") *and* learn a good policy from that structure — with a fixed-size ES search budget, the
  first job can eat most of it. `games/snake.py`'s original 100-float grid observation trained to
  best_fitness 0.65 over 150 generations (`notebooks/0005-neuroevolution-snake.ipynb`) and, per that
  notebook's own diversity-vs-plateau analysis, hit a *representation* ceiling, not a compute one --
  confirmed by swapping to 11 hand-engineered features (danger/heading/food-direction) with the same
  generation-class budget: best_fitness 17.28, and the policy actually eats food instead of dying
  almost immediately. Hand a small evolved/RL network the structure a human would notice at a
  glance, don't make it re-derive that structure from pixels/cells as a side effect of the main task.
- Match-outcome fitness (win/draw/loss per opponent, `evolve.match.MatchFitnessEvaluator`) is a much
  coarser signal than a continuous per-step reward, and gets noisier still when the opponent itself
  has randomness in it (a `rng.choice()`-based strategy) — one match's outcome partly reflects the
  opponent's luck, not just the genome's quality. First `games/checkers.py` evolutionary smoke test
  (2 opponents × 2 seats = 4 fitness values per genome) showed no clean trend over 25 generations;
  the *same* setup with one randomized opponent repeated 6× in the opponent pool (6 independent
  samples of that opponent's luck, still 1 fitness value per match) showed a clear one
  (mean fitness ~0.09 → ~0.21 over 40 generations). If evolution against a match-outcome fitness
  looks stuck, check whether there's enough independent opponent sampling before concluding the
  representation or algorithm is at fault.
- A multi-agent `Strategy` is `(observation, legal_moves) -> move` and can't see the environment, so anything that looks ahead (`simulate()`, a copy + `step()`) or scores each move's resulting position must be an env-bound `StrategyFactory` — a fitness evaluator builds a fresh env per match, so closing over one shared `env` (fine in a single-game script) silently breaks there. Use `MatchFitnessEvaluator(..., env_aware=True)`.
- Training fitness is not performance. It's shaped reward on the seeds the population trained on,
  so it overstates: the best Snake champion scores 16–19 on its 5 training seeds but 11.9 on 200
  held-out ones, and a 3-line greedy heuristic beats it (18.5). Judge a policy with
  `jobs/evaluate.py` (held-out seeds, game score, baselines) before claiming it learned anything
  (docs/design/0007).
- A fixed handful of training seeds is a dataset of that size, and evolution overfits it like any
  learner: with Snake's 5 fixed seeds the champion's unseen-game score peaked by generation ~20 and
  then *fell* while training fitness kept climbing. Record a held-out curve during training
  (`--held-out-every`) and prefer fresh seeds per generation (`--seeds resample:N`, jobs/seeding.py).
- One training run is an anecdote; comparing algorithms takes seeds. Snake runs that differ only in rng seed
  span several points of held-out score, so "A beat B" from one run each means little. Use
  `jobs/snake_experiment.py` (arms × >=5 seeds, identical budgets, tagged runs, final champion scored on the
  held-out games) and report an exact permutation test, not just overlapping ranges -- with 5 seeds, +2.0 points
  was p=0.056 (suggestive) and +3.8 was p=0.008. Include a control that changes *one* thing at a time: NEAT vs.
  lexicase-selected neuroevolution confounds the algorithm with the selection rule until a tournament-selected
  arm is added (docs/design/0008). Five seeds can't measure a *rare* failure: DQN without a target network diverged
  once in five seeds per arm, which reads as either a fluke or a 20% rate; twenty seeds made it ~5% (0 of 45 with
  one). When an arm's spread is one outlier, add seeds before drawing the conclusion.
- **A value-based learner is only as good as the observation is Markov.** Tabular Q-learning on Snake's
  `features.v1` converged (all 256 reachable rows, 5M steps no better than 1M) to ~18 points while NEAT, reading the
  same 11 features, reaches 38: a 38-point policy exists in that table's own space. Different situations share a row
  (the features alias them), so each row's value averages futures that differ and bootstrapping propagates the
  blur; evolution scores whole policies and doesn't care. Before blaming the algorithm's hyperparameters (none of 11
  variants mattered, docs/design/0010), check whether the observation can tell apart the states that need different
  actions -- the same question as the Reach1D entry above, asked of values rather than policies. Confirmed from the
  other side: a DQN (which generalizes, unlike the table) on the same features also stops at ~19, and on
  `egocentric.v1`, which tells those situations apart, reaches 28-30.
- An algorithm with a hidden internal mechanism must report it, or a misconfigured mechanism is
  invisible. For a bootstrapped value learner that's the level of its estimates (`q_mean` in a DQN run's extras): a
  run that diverged (Q = 1.3e10) and one that never learned both score ~0 held out -- only that curve tells them apart. NEAT's speciation sorts genomes by a distance normalized by gene count once a genome
  has >= 20 genes; Snake's 36-gene starting genomes therefore barely differ from each other, and the
  paper's fixed threshold (3.0) left the *species count at 1* in every generation of the first trial
  run -- NEAT quietly degenerating into fixed-topology neuroevolution with a growing network. No test
  failed and best fitness still rose; only the per-generation species count showed it. Fixed by an
  adaptive threshold (`NeatConfig.target_species`), and `evolve_neat` now reports species count and
  champion size through `GenerationStats.extras` so the run page charts them (docs/design/0008).

- A native port of a numeric kernel is only a drop-in if it accumulates in the *same order*: `bias + sum(w*a)`
  and `((bias + w0*a0) + w1*a1)...` differ in the last bits, and an argmax policy or a tie-broken search can
  flip on that. `rust/core/src/nets.rs` follows the Python forward passes' order exactly, so a native Snake
  rollout scores a genome bit for bit like the Python loop (`jobs/tests/test_snake_rollout.py` compares with
  `==`, not a tolerance). Profile before porting: the Snake game already ran in Rust, yet ~87% of a generation
  was the Python forward pass.

## Web / API (`apis/backend`)

- Reuse an existing pydantic model as a FastAPI `response_model`/return type directly (e.g.
  `telemetry.RunInfo`, `telemetry.GenerationStats` in `apis/backend/src/backend/routers/runs.py`)
  instead of redefining an API-layer schema that just mirrors it — one source of truth for the
  shape, one place to add a field.
- Type FastAPI dependency-injected params against the `Protocol` (`RunRegistry`/`MetricsSource`/
  `ArtifactStore`), not the concrete `Sqlite*`/`File*` class — only `dependencies.py`'s factory
  functions should know which concrete backend is in use.
- Adapting a synchronous, blocking, never-returning generator (e.g.
  `FileMetricsStore.subscribe()`, which polls via `time.sleep`) to an async SSE stream: offload each
  `next()` call with `anyio.to_thread.run_sync(next, iterator, abandon_on_cancel=True)` rather than
  running the sync generator directly on the event loop — otherwise its `time.sleep` blocks every
  other request. `abandon_on_cancel=True` (renamed from `cancellable=` in AnyIO 4.1+) lets a client
  disconnect interrupt promptly instead of waiting out a full poll interval.
- **Don't test a live-tailing SSE endpoint by driving it end-to-end through `TestClient`/`app.routes`
  introspection assumptions** — two real gotchas hit building `apis/backend`: (1) Starlette's
  `TestClient` doesn't reliably simulate a mid-stream client disconnect, so a full request against
  an endpoint that never terminates on its own hangs forever waiting for data that isn't coming; (2)
  recent FastAPI versions don't eagerly flatten `include_router()`'s routes into `app.routes` (an
  internal `_IncludedRouter` wrapper defers it), so route-registration checks must go through
  `app.openapi()["paths"][...]` instead of iterating `app.routes`. Test the tricky async-adaptation
  logic directly (call the async generator function, `async for` a bounded number of items, then
  `await gen.aclose()` under `anyio.fail_after(...)` to confirm cancellation is clean) rather than
  through the full ASGI stack. See `apis/backend/tests/test_runs.py`.
- A dict with non-string keys (e.g. `games.snake.Snake.render_state()`'s `cells: dict[(x, y): str]`)
  is not valid JSON and will fail to serialize once it reaches a pydantic `dict[str, Any]` response
  field — Python's tuple keys aren't coercible the way int/float/bool/None keys are. Convert at the
  API boundary (flatten it to a list of `{x, y, label}` records), not in the game library itself — `render_state()`'s shape is also consumed by
  `games.rendering.render_grid_ascii()`, which wants the dict form. (It bit the old Pyodide bridge
  too, as a `ConversionError` in `toJs()` -- any language boundary, same fix: flatten before crossing.)
- **`app.dependency_overrides[dep] = lambda: SomeStore()` creates a *new* instance on every
  request**, not once per test — FastAPI calls the override callable fresh per resolution the same
  way it would the real dependency. For in-memory state that must persist *across* requests within
  one test (e.g. a game session created in one POST and read back in the next), the override must
  close over a single instance created once (`store = SomeStore(); app.dependency_overrides[dep] =
  lambda: store`), not construct one inside the lambda. Silent symptom: a resource created in one
  request 404s in the next, as if it never existed.
- **A Nitro server route resolving a filesystem path via `import.meta.url` isn't reliable once it's
  nested in a subdirectory** — Nitro's dev bundler doesn't preserve a nested route's source-tree depth
  (a since-removed route landed two segments off, not one). Resolve from `process.cwd()` (`nuxt dev`
  and the built server both run from `apps/frontend`), and run it before trusting a path.

- **Train on the horizon you are judged on.** Snake trained on 200-step games and was scored on 1000-step games;
  nothing rewarded surviving longer, and NEAT sat at ~20.6 however many generations or individuals it got (4x
  either: +1). Training on 1000-step games alone gave 31.0, with a larger population and resampled games 38.0
  (`snake-long-v1`, docs/design/0008). Before buying compute, check the training episode matches the evaluation one.
- **Deterministic players make a tiny fitness set.** Checkers strategies that don't randomize replay the same game from
  the standard start, so "3 opponents x 2 seats" is six games however many you list, and `--resample` only reseeds
  tie-breaks. Random openings (`--opening-plies`, both seats sharing one) help the signal but did not by themselves
  make evolution learn. Print `champion gen0 == final` as a first check on any long run: six 300-600 generation
  Checkers runs never left generation 0.
- **A regression target only teaches what its labels contain.** Evolving an evaluator to predict a deeper *material*
  search reached MSE 0.06 yet played far worse than exact material (an approximate leaf evaluator is a worse leaf
  evaluator); playout-outcome labels were too noisy in 300 generations. Distillation needs labels with information the
  student lacks.
- **A shared page/component contract is one interface, not per-game `if`s.** The game page is one
  `GamePage.vue` driven by a `GameModule` (`app/games/types.ts`: score format, extra columns, stage
  components, copy); a second game found the first game's assumptions (a score that's an integer count,
  ONNX model availability, a human "best score") baked into the page, and each became a module field or an
  optional hook. When a new game needs something the page hardcodes, add a module field -- don't fork the
  page.
- **If a run's champion never changes, selection isn't working -- check that before tuning anything else.**
  A checkers run showed a held-out score frozen at its generation-0 value for 149 generations while training
  fitness sat at 0.91; the champion's weights were byte-identical throughout. Three separate causes stacked
  up, each found by measuring: (1) mutating *every* weight of a 500-weight network (`GaussianMutation`'s
  default) wrecks a good parent faster than it can improve it -- mean fitness sat at -0.4; use a small
  per-weight `rate`; (2) a coarse win/draw/loss fitness gives selection nothing to climb when most games are
  draws -- score the margin; (3) opponents seeded by index replay the *same games* every generation, so
  training fitness climbs to +1 while unseen games get worse -- resample per generation (identical within a
  generation, so genomes are still compared fairly).
- **An exception thrown inside a watcher (or any pre-render hook) doesn't just log -- it aborts that component's
  update and leaves Vue's renderer inconsistent, after which *unrelated* components stop updating and the console
  fills with `Cannot read properties of null (reading 'emitsOptions')`.** That is the signature to recognise: a page
  where part of the UI freezes while the rest keeps going. The *first* error in the log is the cause; the cascade
  after it is noise (find it by capturing `unhandledrejection` stacks and looking at the earliest, not by counting
  errors). Here it was an unguarded `path.at(-1)![0]` in the piece-motion watcher. Wrap anything decorative
  (animation planning) in try/catch that falls back to plain state, and guard index access on possibly-empty arrays.
- **A component file created while `nuxt dev` is running may not be registered**: it renders as an unknown
  lowercase custom element (`<boardresult>`), silently, with no Vue warning. Restart the dev server after adding
  components; if a new component "does nothing", look for the unresolved tag in the DOM before debugging its logic.
- **Animating a keyed list: render in a stable order, or Vue will move the DOM nodes.** If a list is keyed (so each
  item is one element) but *rendered in the data's order*, an item whose position in the data changes -- a moved
  piece an engine lists last -- makes Vue physically re-insert its node, and re-inserting restarts its CSS: the
  transition glide is lost and the entrance animation replays (with its staggered delay, the item is invisible
  meanwhile). Sort by a stable key. And when the *scene* changes (a new game), change the keys too, or the
  survivors keep their elements and glide across from wherever the last scene left them. Measure it with a
  `MutationObserver` counting inserted nodes during ordinary play (should be ~0), not by eye.
- **A hidden browser pane freezes CSS animations and transitions** (the tool says "Browser pane is currently
  hidden"): every animated element sits on its first frame -- opacity 0, un-moved -- and any test reading
  *computed* styles reports nonsense. Test animation *logic* through what doesn't depend on the clock (the `style`
  attribute a component set, node identity, DOM mutations), and be honest that timing/visuals were not seen.
- **A live panel must not change size as its contents change.** Anything that updates every game step (a
  status line, a candidate list, a move log, a hint that appears on some plies) needs a fixed or reserved
  height, single-line truncation with a `title` for long text, and its own slot per player rather than one
  panel that swaps between them -- otherwise each ply re-flows the page. Verify by sampling the board's
  top offset and `scrollHeight` across many plies (they should each take one value), not by eye.
- **Nuxt/Vue gotchas that only show at runtime** (`pnpm typecheck` passes on all of them): (0) an *absent*
  boolean prop is `false`, not `undefined` -- an on-by-default prop (`diagnostics`, `showPlayers`) needs
  `withDefaults`, or `x !== false` silently hides it; (1) an imported
  type can't be the *whole* props type (`defineProps<StageProps>()` fails to compile; spell members
  out inline, imported types inside members are fine); (2) in a plain composable, start every
  `useAsyncData` *before* the first `await` -- unlike `<script setup>`, it doesn't preserve Nuxt's context
  across awaits ("composable called outside of a plugin..."); (3) files outside `composables/`/`utils/`
  aren't auto-imported (`app/games/*`): import them explicitly; (4) `nuxt.config.ts` is serialized into the
  build, so a *function* in `app.head` (a `titleTemplate` callback) is silently dropped -- every page's title
  lost its " · Red Queen" suffix until `pnpm typecheck` flagged it. Put it in `app.vue`'s `useHead()`.
- **Never fill a reactive collection one response at a time, and don't make bulk data deeply reactive.** The runs page
  fetched 115 histories in parallel and did `histories.value = { ...histories.value, [id]: h }` per response into a
  deep `ref`: every response re-ran the page's row computation and re-rendered the table (115 x 115), and Vue proxied
  31K records -- 4.5 s of blocked main thread on load (36 s on a refresh), clicks ignored meanwhile, while the network
  and JSON parsing took 0.37 s together. Collect with `Promise.all`, assign once, and hold data you only read in a
  `shallowRef` + `markRaw`. Better still, don't ship a page data it only summarizes (`GET /runs/summaries`).
  Measure with a `longtask` `PerformanceObserver` in a *fresh* tab before and after, not by feel.

## Client-side inference (docs/design/0009)

- **Check an exported model by the decisions it makes, not only its numeric error.** Every evolved
  Snake champion exported to float32 was within ~1e-6 of its float64 original, yet five of ten chose a
  different move somewhere in 200 held-out games (an old grid champion on 11% of moves): argmax on
  near-ties flips. `jobs/publish_models.py` measures action agreement on the protocol's games; a
  variant that disagrees is ranked as its own entrant. The LM equivalent is top-1 next-token agreement,
  measured the way a client generates (int8's dynamic quantization picks activation scales per call,
  so one-token-at-a-time decoding legitimately differs from a full pass).
- **Exact ties exist.** A saturated `tanh` returns exactly ±1.0, so outputs can tie exactly, and two
  correct `tanh` implementations one ulp apart then break the tie differently (fp64 ONNX still
  disagreed with Python on 0.5% of one champion's moves). Agreement below 100% isn't always a bug —
  but it always has to be reported.
- **Porting a game: order is behaviour.** Strategies index into `legal_moves()`, and the Python
  Checkers generated moves in dict insertion order (a moved piece goes last) — the Rust board had to
  reproduce dict semantics, not just the rules. Keep the original as an oracle
  (`libs/games/tests/reference_*.py`) and compare whole trajectories, not samples of positions.
- **ONNX Runtime Web needs shape tensors inside the graph.** It resolves a `Reshape`'s target shape
  before attaching external data, so sharding an int64 constant fails session creation — in the
  browser only (the Python runtime inlines everything first). `modelpack` keeps int32/int64 and
  < 1 KB tensors inline and shards the rest (int8 weights included).
- **A GPU isn't automatically faster.** Batch-1 decoding is dispatch-bound: the 79K-parameter TinyLM
  ran ~15x faster on WASM than WebGPU, and at 85M parameters fp32-on-WebGPU and int8-on-WASM tied
  (~83 tokens/s). Small packages list WASM first; measure before assuming a backend.

## Lessons

Freeform notes that don't fit a section above yet. Once a pattern shows up twice, fold it into the
relevant section instead of leaving it here.

- The C++ bugs above were all found by actually building and running the code, not by reading it —
  verify an implementation runs before trusting that it looks correct.
- `apps/frontend` pages under a transitioned `<NuxtPage>` need exactly **one** template root — a
  sibling HTML comment counts as a second root in dev and silently blanks the page on client-side
  navigation (Nuxt warns `NUXT_E4004` in the console; direct loads look fine). Found by clicking
  through, not by loading each page directly.
- **A server-rendered `apps/frontend` page must not write unrounded `Math.tanh` (or other libm) results
  into an attribute or style.** Node and the browser can disagree in the last digit
  (`width:10.368710750128162%` on the server vs `...157%` on the client) and Vue reports a hydration
  mismatch. Round to ~3 decimals wherever an activation becomes a style/attribute string
  (`XorTable`, `NeatDiagram`, `NetworkDiagram`). Only surfaced by loading a chapter in a *fresh* browser
  tab and reading its console -- the accumulated console of a long-lived tab hides which page an error
  came from.
- **A hidden browser (a background tab, a collapsed in-app browser pane) never finishes a route change.**
  `requestAnimationFrame` doesn't fire while `document.visibilityState === "hidden"`, and the site's
  `out-in` page transition waits on it: the URL changes, the old page stays, and Vue logs
  `Cannot read properties of null (reading 'Symbol(_leaveCb)')`. Not a site bug; verify navigation
  with the page visible, or load the target URL directly.
- `array.map(fn)` passes `(item, index, array)`: handing it a function with an optional second
  parameter silently feeds the index in. `population.map(this.evaluate)` with
  `xorWeightsFitness(weights, layers = XOR_SHAPE)` made `layers` a number, so a Learn demo's evolution
  quietly stalled at 0.625 instead of solving XOR by generation ~15. Wrap it: `map((w) => fn(w))`.
  An offline probe that *did* wrap it passed, which is why the demo had to be driven end to end.
- Snake's `render_state()` cells are ordered head-*last* (`[nearest-head, ..., tail, head]`), so
  "the segment behind the head" is `cells[0]`, not `cells.at(-2)` (that's the tail) — the board's
  eye direction got this wrong until the live network diagram disagreed with it.
