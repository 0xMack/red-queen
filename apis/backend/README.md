# backend

FastAPI service exposing `libs/telemetry` and `libs/games` to `apps/frontend`. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and the incremental plan this implements.

## Contents

- `routers/runs.py` — run/metrics/artifact endpoints, reusing `telemetry.RunInfo` and
  `telemetry.GenerationStats` directly as response models (no duplicate API-layer schemas):
  - `GET /runs` — `list[RunInfo]`
  - `GET /runs/{run_id}` — `RunInfo` (404 if unknown)
  - `GET /runs/{run_id}/metrics/history?since_generation=N` — `list[GenerationStats]`
  - `GET /runs/{run_id}/metrics/stream?since_generation=N` — SSE of `GenerationStats`,
    backfill-then-live. Adapts `telemetry.FileMetricsStore.subscribe()` (a synchronous,
    never-returning, polling generator) to async by offloading each `next()` call to a worker
    thread (`anyio.to_thread.run_sync(..., abandon_on_cancel=True)`) so a blocking poll doesn't
    stall the event loop, and so a client disconnect cancels promptly instead of waiting out a
    full poll interval.
  - `GET /runs/{run_id}/artifacts/{ref}?kind=program|trace` — raw bytes (404 if unknown). Note
    `ArtifactStore` itself isn't run-scoped (it's a flat `ref -> bytes` store); `run_id` stays in
    the URL for REST grouping, matching doc 0005's endpoint table.
- `routers/games.py` — game session endpoints. The session's `Environment` (e.g. `games.Snake`)
  runs server-side in `game_sessions.GameSessionStore`, an in-memory, single-process registry with
  no telemetry/persistence layer (a watch/play session isn't a training run):
  - `POST /games/{game}/sessions` — create a session (`GameSessionCreate`: `game`, optional `seed`)
    → `GameSessionState` with the initial `render_state`. 404 for an unregistered game, 400 if the
    path and body `game` disagree.
  - `POST /games/{game}/sessions/{id}/actions` — apply one action (`ActionRequest.action: float`)
    → the resulting `GameSessionState`. 404 for an unknown session, 409 once `done`.
  - `GET /games/{game}/sessions/{id}/trajectory` — the full episode so far as a `TrajectoryArtifact`
    (states/actions/rewards).
  - `render_state()`'s `cells` (grid games) is `dict[(x, y): label]` — not valid JSON as-is (tuple
    keys). `game_sessions.json_safe_render_state()` flattens it to `[{x, y, label}, ...]` before it
    ever reaches a response model.
  - Only games implementing `games.rendering.Renderable` are registered (`games.snake.Snake` today;
    `games.reach1d.ReachTarget1D` doesn't have a `render_state()` yet, so it isn't playable through
    this API).
- `dependencies.py` — FastAPI providers. Telemetry-backed ones use the **Protocols**
  (`RunRegistry`/`MetricsSource`/`ArtifactStore`), backed by the `Sqlite*`/`File*` implementations
  pointed at `settings.run_data_dir()`, so a storage backend swap never touches router code.
  `GameSessionStore` has no such Protocol — it's in-memory, single-implementation, app-specific
  state, not something `docs/design/0002`'s swappable-backend story applies to.
- `settings.py` — locates the run-data directory (`jobs/run-data` by default, matching
  `jobs/baseline_gp_run.py`'s `RUN_DATA_DIR`; override with `REDQUEEN_RUN_DATA_DIR`).

## Running it

From the repo root:

```
uv run uvicorn backend.main:app --app-dir apis/backend/src --reload
```

Then, having run `uv run python jobs/baseline_gp_run.py` at least once so there's real data to
serve: `http://127.0.0.1:8000/runs`.

## Testing

```
uv run pytest apis/backend/tests
```

Tests override the `dependencies.py` providers with `tmp_path`-backed telemetry instances (same
pattern as `libs/telemetry/tests`), so they don't touch `jobs/run-data`. `test_games.py`'s `client`
fixture overrides `_game_session_store` with a single `GameSessionStore()` instance shared across
every request *within* one test (not a fresh one per request) — a per-request lambda would make a
session created in one call invisible to the next, since nothing else keeps it alive.
