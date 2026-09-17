# jobs

Short- or long-running jobs, tasks, and workers — e.g. training runs, batch evaluations, or
background simulation workers. This is where algorithm libs (which never import `telemetry`
themselves — see docs/design/0001) get wired to it for a real run.

## Contents

- `baseline_gp_run.py` — runs `libs/evolve`'s baseline GA loop against the fixed benchmark problem
  (docs/design/0003 phase 1), recording every generation to a local `telemetry` store
  (`run-data/`, gitignored) and reading it back to prove the whole evolve → metrics → replay
  pipeline works. Run with `uv run python jobs/baseline_gp_run.py` from the repo root.
