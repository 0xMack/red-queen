# backend

FastAPI service exposing `libs/telemetry` and `libs/games` to `apps/frontend`. See
[../../docs/design/0005-frontend-and-api-contracts.md](../../docs/design/0005-frontend-and-api-contracts.md)
for the full contract and the incremental plan this implements.

## Contents

- `routers/runs.py` — run/metrics/artifact endpoints, reusing `telemetry.RunInfo` and
  `telemetry.GenerationStats` directly as response models (no duplicate API-layer schemas):
  - `GET /runs` — `list[RunInfo]`
  - `GET /runs/summaries?trend_points=60` — `list[RunSummary]`: per run, generations recorded, best fitness,
    the last `GenerationStats` and a downsampled best-fitness trend. What a runs list needs in one small request
    (164 KB for 115 runs) instead of every run's full history (13 MB); `FileMetricsStore.history` parses each
    appended line once and keeps it, so repeated calls cost milliseconds. Declared before `/{run_id}`.
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
  - `POST /runs/{run_id}/control` — `ControlRequest{action: pause|resume|step}` → 202. `pause`/
    `resume` just flip the run's `RunStatus` via `RunRegistry.update_status()` — that status field
    already existed (docs/design/0002) and is the *only* coordination between this endpoint and a
    (separate-process) training job's `jobs/control.py` callback, which blocks while status is
    `"paused"`. `step` is driven entirely from this side: resume, poll `MetricsSource.history()`
    until exactly one new generation is recorded (504 after `timeout_s`, default 30s), then
    re-pause — deliberately *not* a third `RunStatus` value, so the job-side callback only ever
    needs to understand two states.
- There is no game-session API any more: every game is played client-side (docs/design/0009 -- the Rust core
  compiled to WebAssembly), so the server-side sessions docs/design/0005 step 4 started with were removed.
- `dependencies.py` — FastAPI providers. Telemetry-backed ones use the **Protocols**
  (`RunRegistry`/`MetricsSource`/`ArtifactStore`), backed by the `Sqlite*`/`File*` implementations
  pointed at `settings.run_data_dir()`, so a storage backend swap never touches router code.
- `settings.py` — locates the run-data directory (`jobs/run-data` by default, matching
  `jobs/run_context.py`'s `RUN_DATA_DIR`; override with `REDQUEEN_RUN_DATA_DIR`).

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
pattern as `libs/telemetry/tests`), so they don't touch `jobs/run-data`.
