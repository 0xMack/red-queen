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
    backfill-then-live. Polls `history(since_generation=next)` (incremental: it parses only what
    was appended) in a worker thread, and waits between polls with `anyio.sleep` on the event loop,
    where a client disconnect or a shutdown cancels it. It deliberately doesn't wrap the store's
    `subscribe()`: that generator's `next()` blocks until a generation arrives -- for a finished
    run, never -- so each abandoned call parked a non-daemon anyio worker thread forever, and a
    `--reload` then waited on it and never started the new server.
  - `GET /runs/{run_id}/artifacts/{ref}?kind=program|trace` — raw bytes (404 if unknown). Note
    `ArtifactStore` itself isn't run-scoped (it's a flat `ref -> bytes` store); `run_id` stays in
    the URL for REST grouping, matching doc 0005's endpoint table.
  - `POST /runs/{run_id}/control` — `ControlRequest{action: pause|resume|step}` → 202. `pause`/
    `resume` just flip the run's `RunStatus` via `RunRegistry.update_status()` — that status field
    already existed (docs/design/0002) and is the *only* coordination between this endpoint and a
    (separate-process) training job's `jobcore.control` callback, which blocks while status is
    `"paused"`. `step` is driven entirely from this side: resume, poll `MetricsSource.history()`
    until exactly one new generation is recorded (504 after `timeout_s`, default 30s), then
    re-pause — deliberately *not* a third `RunStatus` value, so the job-side callback only ever
    needs to understand two states.
  - **The write side** (docs/design/0018: what a job on the HTTP sink, `jobcore.http`, writes through -- a worker
    that can't share the data directory): `POST /runs` (`RunCreate{config}` → 201 `{run_id}`), `PATCH /runs/{run_id}`
    (`RunPatch{status?, summary?, config?}`; `config` is *merged*, a value of null removing a key, for corrections
    after the fact) → `RunInfo`, `POST /runs/{run_id}/metrics` (a list of `GenerationStats`, appended in order; 422
    if any belongs to another run, and then none is recorded), `PUT /runs/{run_id}/artifacts/{ref}?kind=` (the raw
    body; the run segment is ignored, as on GET). Golden runs recorded over this API match the local sink's exactly
    (`jobs/tests/test_http_sink.py`).
- There is no game-session API any more: every game is played client-side (docs/design/0009 -- the Rust core
  compiled to WebAssembly), so the server-side sessions docs/design/0005 step 4 started with were removed.
- `routers/leaderboards.py` — `GET /games/{game}/leaderboard?protocol=` (every `EvaluationRecord`, best mean first) and
  `GET /games/{game}/interfaces`; and, for the evaluator on the HTTP sink, `PUT /games/{game}/evaluations` (one record,
  replacing that entrant's under its protocol; 422 if it's another game's) and `POST /games/{game}/evaluations/prune`
  (`{protocol, keep}` → the entrant ids removed).
- `routers/models.py` — the local model store's blobs, manifests and per-game catalogs, and on-demand packaging of an
  unpublished champion (docs/design/0009).
- `routers/jobs.py` + `services/` — jobs, experiments and spec schemas (docs/design/0018); the API records, the
  scheduler runs:
  - `POST /jobs` — a train/evaluate/publish spec (`kind` defaults to `train`) → 201 `Job` (`queued`); 422 with the
    reason if it doesn't validate *and resolve*, so the scheduler is never handed something that can't run.
  - `GET /jobs?status=&kind=&parent=`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` (queued: cancelled at once;
    running: `cancel_requested`, for the scheduler to stop the worker).
  - `GET /jobs/{id}/logs?offset=N` — `{offset, text, finished}`: the worker's log from byte N, and where to ask
    next. A client follows a running job by polling it.
  - `POST /experiments` — an `ExperimentSpec` (stored; resubmitting resumes) → its status. `GET /experiments`,
    `GET /experiments/{name}` — the arm × seed matrix: `completed` (a completed run tagged with the experiment, arm
    and seed, however it was trained), a job's status, or `pending`; `done`/`total`, baselines, whether a report
    exists. `GET /experiments/{name}/report` — `<data dir>/experiments/<name>.json`.
  - `GET /specs/schemas` — JSON Schema of every payload kind and of each algorithm's, protocol's and target's params.
- `dependencies.py` — FastAPI providers. Telemetry-backed ones use the **Protocols**
  (`RunRegistry`/`MetricsSource`/`ArtifactStore`), backed by the `Sqlite*`/`File*` implementations
  opened once by `telemetry.open_stores()`, so a storage backend swap never touches router code.
- `settings.py` — the backend's own settings (`REDQUEEN_MODELS_BASE_URL`). Where the data lives is not one of them:
  `dependencies.py` opens the stores with `telemetry.open_stores()`, the same call every job makes, so the data
  directory (`REDQUEEN_DATA_DIR`, default `<repo>/data`) is defined once (docs/design/0018).

## Running it

From the repo root:

```
uv run uvicorn backend.main:app --app-dir apis/backend/src --reload --timeout-graceful-shutdown 3
```

Then, having run `uv run python -m trainer jobs/trainer/specs/gp.yaml` at least once so there's real data to
serve: `http://127.0.0.1:8000/runs`.

## Testing

```
uv run pytest apis/backend/tests
```

Tests override the `dependencies.py` providers with `tmp_path`-backed telemetry instances (same
pattern as `libs/telemetry/tests`), so they don't touch `data/`.
