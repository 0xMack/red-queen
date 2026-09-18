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
- **`ruff format`/`ruff check --fix` a specific set of files you actually touched, not a whole
  directory** — no `line-length` is configured anywhere in this repo (checked: every `pyproject.toml`
  omits it, so `ruff format` uses its default, 88), but plenty of pre-existing hand-written code
  runs longer than that. Formatting a whole directory reflows every one of those files as a side
  effect, mixing unrelated cosmetic churn into an otherwise-scoped commit. Happened while building
  docs/design/0005 step 6: `ruff format libs/evolve` reformatted six files this change never
  touched. Fixed by reverting those and re-running ruff against only the files actually part of the
  change.

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
  API boundary (`game_sessions.json_safe_render_state()` flattens it to a list of `{x, y, label}`
  records), not in the game library itself — `render_state()`'s shape is also consumed by
  `games.rendering.render_grid_ascii()`, which wants the dict form. **The same underlying problem
  recurs in Pyodide, with a different error**: `pyProxy.toJs()` raises `pyodide.ffi.ConversionError:
  Cannot use (x, y) as a key for a Javascript Map` on the exact same tuple-keyed dict — a JS `Map`
  can technically hold any key, but Pyodide's converter explicitly refuses non-primitive ones. Same
  fix, different side of the boundary: flatten on the Python side before it crosses into JS (see
  `app/workers/snakeGame.worker.ts`'s `render_state_for_js`), not by trying to coax
  `toJs()`/`dict_converter` into accepting tuple keys.
- **`app.dependency_overrides[dep] = lambda: SomeStore()` creates a *new* instance on every
  request**, not once per test — FastAPI calls the override callable fresh per resolution the same
  way it would the real dependency. For in-memory state that must persist *across* requests within
  one test (e.g. a game session created in one POST and read back in the next), the override must
  close over a single instance created once (`store = SomeStore(); app.dependency_overrides[dep] =
  lambda: store`), not construct one inside the lambda. Silent symptom: a resource created in one
  request 404s in the next, as if it never existed. See `apis/backend/tests/test_games.py`.
- **A Nitro server route resolving a filesystem path via `import.meta.url` isn't reliable once it's
  nested in a subdirectory.** `apps/frontend/server/api/py-games.get.ts` (flat, one directory under
  `server/`) resolved a `libs/` path correctly with `new URL("../../../../libs/...",
  import.meta.url)`. Moving the equivalent logic into
  `server/api/py-source/[pkg].get.ts` (one directory deeper) and adding one more `../` — the
  "obviously correct" fix by source-tree depth — landed on a path missing *two* segments, not the
  expected one; Nitro's dev-mode bundler doesn't preserve this route's real source-tree depth for
  nested routes. Found by actually running it (`ENOENT`, then inspecting the resolved path), not by
  reasoning about it. Fixed by resolving from `process.cwd()` instead (reliable because `nuxt dev`
  and the built server are both always run from `apps/frontend`) — don't trust
  `import.meta.url`-relative depth counting for a Nitro route without running it.

## Lessons

Freeform notes that don't fit a section above yet. Once a pattern shows up twice, fold it into the
relevant section instead of leaving it here.

- The C++ bugs above were all found by actually building and running the code, not by reading it —
  verify an implementation runs before trusting that it looks correct.
- `apps/frontend` pages under a transitioned `<NuxtPage>` need exactly **one** template root — a
  sibling HTML comment counts as a second root in dev and silently blanks the page on client-side
  navigation (Nuxt warns `NUXT_E4004` in the console; direct loads look fine). Found by clicking
  through, not by loading each page directly.
- Snake's `render_state()` cells are ordered head-*last* (`[nearest-head, ..., tail, head]`), so
  "the segment behind the head" is `cells[0]`, not `cells.at(-2)` (that's the tail) — the board's
  eye direction got this wrong until the live network diagram disagreed with it.
