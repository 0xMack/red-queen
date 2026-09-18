# Coding Guidelines

Standards for this repo, plus accumulated lessons (see [AGENTS.md](../AGENTS.md) for the bar on
what belongs here and how to add to it). Read before writing code, not after.

## General

- Keep abstractions pluggable only at proven extension points (`Program`, `FitnessEvaluator`,
  `SelectionStrategy`, `VariationStrategy` per docs/design/0001) — don't invent new ones
  speculatively.
- Dependency direction: algorithm/lib packages never import `telemetry`; only integration glue
  (usually in `jobs/`) imports both. See docs/design/0002.

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
  `usePyodideGames.ts`'s `render_state_for_js`), not by trying to coax `toJs()`/`dict_converter`
  into accepting tuple keys.
- **`app.dependency_overrides[dep] = lambda: SomeStore()` creates a *new* instance on every
  request**, not once per test — FastAPI calls the override callable fresh per resolution the same
  way it would the real dependency. For in-memory state that must persist *across* requests within
  one test (e.g. a game session created in one POST and read back in the next), the override must
  close over a single instance created once (`store = SomeStore(); app.dependency_overrides[dep] =
  lambda: store`), not construct one inside the lambda. Silent symptom: a resource created in one
  request 404s in the next, as if it never existed. See `apis/backend/tests/test_games.py`.

## Lessons

Freeform notes that don't fit a section above yet. Once a pattern shows up twice, fold it into the
relevant section instead of leaving it here.

- The C++ bugs above were all found by actually building and running the code, not by reading it —
  verify an implementation runs before trusting that it looks correct.
