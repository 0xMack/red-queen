# jobs

Short- or long-running jobs, tasks, and workers — e.g. training runs, batch evaluations, or
background simulation workers. This is where algorithm libs (which never import `telemetry`
themselves — see docs/design/0001) get wired to it for a real run.

## Contents

- `baseline_gp_run.py` — runs `libs/evolve`'s baseline GA loop against the fixed benchmark problem
  (docs/design/0003 phase 1), recording every generation to a local `telemetry` store
  (`run-data/`, gitignored) and reading it back to prove the whole evolve → metrics → replay
  pipeline works. Run with `uv run python jobs/baseline_gp_run.py` from the repo root.
- `control.py` — `make_control_callback(registry, run_id)`, an `on_generation` callback that blocks
  while `RunRegistry` reports the run's status as `"paused"` (docs/design/0005's control API,
  step 7). This is the only coordination needed between `apis/backend`'s `POST
  /runs/{id}/control` endpoint (a separate process) and a running job: both just read/write the
  same `RunRegistry` row, reusing `RunStatus`'s existing `"paused"` value rather than adding new
  shared state. `baseline_gp_run.py` wires it in alongside its telemetry callback.

`jobs/` isn't a `uv` workspace package (no `pyproject.toml`) — scripts here import already-installed
workspace packages, and `uv run python jobs/<script>.py` puts the script's own directory on
`sys.path` automatically, which is how e.g. `baseline_gp_run.py` imports `control.py` as a plain
sibling module. `jobs/tests/conftest.py` does the same thing explicitly for `uv run pytest
jobs/tests`.
