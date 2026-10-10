# jobcore

What every workload (docs/design/0018: trainer, evaluator, publisher, scheduler) shares. Small on purpose: pydantic,
`telemetry`, `arena` and `evolve` only, so the backend and the CLI can import the specs without pulling in a trainer.

- `jobcore.sink` — `open_sink()`: where a workload writes. A sink is telemetry's `Stores`, every store typed by its
  Protocol; `local` (the default; `REDQUEEN_SINK`) is the file/SQLite stores under `telemetry.data_dir()`. An HTTP
  sink, for workers that can't share the data directory, arrives with the backend's write endpoints (stage 5).
- `jobcore.lifecycle` — `with recorded_run(config) as run:` creates the run and marks it `completed`, or `failed` if
  the block raises (a crashed run used to stay `running` forever); `run.control_callback(cost)` wires pause/resume
  without counting paused time; `run.set_training_summary(cost)` writes the standard summary.
- `jobcore.control` — `make_control_callback(registry, run_id)`: blocks while the run's status is `paused`, the job
  side of the backend's `POST /runs/{id}/control` (docs/design/0005).
- `jobcore.specs` — `TrainSpec` and the algorithm registry: each algorithm registers a params model and the budget
  units it understands; `spec.resolve()` validates a spec against it; `schemas()` is the JSON Schema of everything.
  The params models live here, not in the trainer, so the backend can validate a spec without importing one.
- `jobcore.parallel` — `ProcessPoolEvaluator`: a generation's genomes scored across processes, reproducing a serial
  run exactly (every fitness evaluator here is deterministic).

Tests: `uv run pytest jobs/core`.
