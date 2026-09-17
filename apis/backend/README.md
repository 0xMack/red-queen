# backend

FastAPI service exposing `libs/telemetry` (and, later, `libs/games`) to `apps/frontend`. See
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
    thread (`anyio.to_thread.run_sync(..., cancellable=True)`) so a blocking poll doesn't stall the
    event loop, and so a client disconnect cancels promptly instead of waiting out a full poll
    interval.
  - `GET /runs/{run_id}/artifacts/{ref}?kind=program|trace` — raw bytes (404 if unknown). Note
    `ArtifactStore` itself isn't run-scoped (it's a flat `ref -> bytes` store); `run_id` stays in
    the URL for REST grouping, matching doc 0005's endpoint table.
- `dependencies.py` — FastAPI providers for `telemetry`'s `RunRegistry`/`MetricsSource`/
  `ArtifactStore` **Protocols**, backed by the `Sqlite*`/`File*` implementations pointed at
  `settings.run_data_dir()`. Routers depend on the Protocol types, so a future storage backend swap
  never touches router code.
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
pattern as `libs/telemetry/tests`), so they don't touch `jobs/run-data`.
