# jobs

Short- or long-running jobs, tasks, and workers — e.g. training runs, batch evaluations, or
background simulation workers. This is where algorithm libs (which never import `telemetry`
themselves — see docs/design/0001) get wired to it for a real run.

## Contents

- `baseline_gp_run.py` — runs `libs/evolve`'s baseline GA loop against the fixed benchmark problem
  (docs/design/0003 phase 1), recording every generation to a local `telemetry` store
  (`run-data/`, gitignored) and reading it back to prove the whole evolve → metrics → replay
  pipeline works. Run with `uv run python jobs/baseline_gp_run.py` from the repo root.
- `snake_neuro_run.py` — the same neuroevolution-against-`games.snake` setup already validated in
  `notebooks/0005-neuroevolution-snake.ipynb`, wired to `telemetry` end to end instead of collecting
  summaries in a plain list. Its champions are real `ArtifactStore` entries, serialized with
  `WeightVector.to_json()` (the real, round-trippable wire format — see `libs/evolve/README.md`),
  which is what makes docs/design/0005 step 6 (watching a trained policy play, in
  `apps/frontend`) possible: the API needed no changes at all to serve them, since
  `GET /runs/{id}/artifacts/{ref}` already returns raw bytes regardless of what's inside. Run with
  `uv run python jobs/snake_neuro_run.py [interface_id] [--seeds fixed:5|fixed:N|resample:N]
  [--held-out-every 10] [--generations 250]` — takes a few minutes (250 generations ×
  100 individuals × 5 benchmark scenarios; about 2.5× longer under the 100-input
  `snake/grid-flat.v1+relative3.v1`). Records `config.interface`, `training_seeds`, `rng_seed`, and
  a measured `summary.cost` block (below). Deterministic: the same interface, seed strategy, and
  `RNG_SEED` reproduce the same champion exactly. Every `--held-out-every` generations it also records
  the champion's game score on `evaluate.MONITOR_SEEDS` (`GenerationStats.held_out_score`) -- the
  curve that shows overfitting.
- `seeding.py` — training-seed strategies: `fixed:N` (the same N games every generation; `fixed:5`
  is the original behavior) or `resample:N` (fresh games every generation from a pool disjoint from
  every evaluation seed range, via a seeded rng). With `fixed:5` the Snake champion memorizes its 5
  games: its unseen-game score peaks around generation 20-25 and then declines while training
  fitness keeps rising.
- `costs.py` — `TrainingCostMeter` (an `on_generation` callback; wrap the control callback with
  `excluding_pauses()` so paused time isn't counted) records exact counters (fitness evaluations,
  episodes, env steps — hardware-independent) plus active/CPU time, peak memory, and a
  `hardware_fingerprint()` (docs/design/0007: counters first, clocks second).
- `evaluate.py` — the leaderboard job (docs/design/0007). Runs every *finished* game run's final
  champion (under its recorded interface) and fixed baselines (random, greedy) on held-out seeds
  under protocol `snake.score.v1`, and writes an `EvaluationRecord` per entrant: score
  distribution with a 95% interval, the train-vs-held-out gap, per-decision inference latency
  (encode vs. decide), parameters, and training cost (measured, or estimated and labelled so for
  pre-cost-tracking runs). Re-run it after new runs finish. Changing seeds/caps/rules means bumping
  `PROTOCOL`, never editing it in place.
- `checkers_round_robin.py` — four static Checkers strategies (random, first-legal, one- and two-ply
  material lookahead) played against each other, 200 games per pairing with seats alternating; the
  numbers the "Multi-Agent Games" Learn chapter cites (headline: one ply of lookahead ≈ random,
  because captures are mandatory; two plies wins ~95%).
- `backfill_interfaces.py` — one-off: sets `config.interface` on game runs recorded before
  interfaces existed, resolved from each champion's own layer sizes (idempotent).
- `control.py` — `make_control_callback(registry, run_id)`, an `on_generation` callback that blocks
  while `RunRegistry` reports the run's status as `"paused"` (docs/design/0005's control API,
  step 7). This is the only coordination needed between `apis/backend`'s `POST
  /runs/{id}/control` endpoint (a separate process) and a running job: both just read/write the
  same `RunRegistry` row, reusing `RunStatus`'s existing `"paused"` value rather than adding new
  shared state. Both `baseline_gp_run.py` and `snake_neuro_run.py` wire it in alongside their
  telemetry callback.

`jobs/` isn't a `uv` workspace package (no `pyproject.toml`) — scripts here import already-installed
workspace packages, and `uv run python jobs/<script>.py` puts the script's own directory on
`sys.path` automatically, which is how e.g. `baseline_gp_run.py` imports `control.py` as a plain
sibling module. `jobs/tests/conftest.py` does the same thing explicitly for `uv run pytest
jobs/tests`.
