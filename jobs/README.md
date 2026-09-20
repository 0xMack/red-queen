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
  `snake/grid-flat.v1+relative3.v1`). Also takes `--selection lexicase|tournament`, `--rng-seed`,
  and `--experiment NAME` (see `snake_experiment.py`). Records `config.interface`, `training_seeds`, `rng_seed`, and
  a measured `summary.cost` block (below). Deterministic: the same interface, seed strategy, and
  `RNG_SEED` reproduce the same champion exactly. Every `--held-out-every` generations it also records
  the champion's game score on `evaluate.MONITOR_SEEDS` (`GenerationStats.held_out_score`) -- the
  curve that shows overfitting.
- `snake_neat_run.py` — NEAT (`libs/evolve/src/evolve/neat.py`, docs/design/0008) against the same
  Snake interface, seed strategies, held-out monitoring, and cost meter as `snake_neuro_run.py` —
  it imports that script's callbacks — so the two are directly comparable. Champions are stored as
  `NeatGenome.to_json()`; per-generation telemetry carries `extras` (species count, the champion's
  hidden nodes and connections, the adaptive compatibility threshold). Run with `uv run python
  jobs/snake_neat_run.py [--seeds resample:5] [--rng-seed 0] [--no-speciation]`; ~1.5 min for 250
  generations. Deterministic per `--rng-seed`.
- `snake_experiment.py` — a tracked *comparison*: arms (`neuro-lexicase`, `neuro-tournament`, `neat`,
  `neat-no-speciation`) × rng seeds, every run recorded to telemetry and tagged `config.experiment`/
  `arm`/`rng_seed`, resumable, arms parallelizable as separate processes. `report --name NAME`
  scores each run's *final* champion on the 200 leaderboard games and writes per-arm aggregates to
  `run-data/experiments/NAME.json` (gitignored — the durable record of a result is
  docs/design/0008 and the Learn chapter). Experiment-tagged runs are deliberately kept off the
  leaderboard (`evaluate.py` skips them) so 20 seeds don't bury every other entrant.
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
- `checkers_training.py` — what the two Checkers training jobs share: a genome (layered or NEAT) becomes a
  player via `strategy_factory(genome, depth)` (alpha-beta to `depth` plies in the Rust core, the network
  scoring the leaves); `OpponentPool` is the fitness evaluator (fixed opponents, optionally a hall of fame of
  the run's own past champions, optionally *resampled every generation* -- fixed opponent games are
  memorized: training fitness reached +1 while games never seen got worse); `margin_scorer` scores a draw by
  the material edge held; `material_seed_*` start part of a population as a noisy material evaluator.
- `checkers_neat_run.py` — the NEAT counterpart of `checkers_neuro_run.py`, same opponents / depth / hall /
  monitor, evolving a graph that starts as a linear evaluator (NEAT's structural mutations grow it) and is
  played through the core's `GraphNet`. Gentler weight mutation than NEAT's default (see the file).
- `checkers_neuro_run.py` — neuroevolution against Checkers, recorded to telemetry: a genome is a
  32→H→1 *position evaluator* (the Rust `evaluator` strategy plays the move whose resulting
  position is worst for the opponent, so a whole match is a few native calls: ~4x faster than scoring
  in Python), fitness is `MatchFitnessEvaluator(env_aware=True)`
  against the static strategies (one value per opponent per seat, lexicase selection), and the
  held-out score is win/draw/loss on games with opponent seeds training never used.
- `evaluate_versus.py` — the two-player leaderboard (docs/design/0007's "Versus"): protocol
  `checkers.versus.v1`, a round robin over every finished checkers champion plus the fixed baselines,
  20 games per pair with seats alternating. An entrant's score is **points per game** (win 1, draw ½)
  against every *other* entrant, with a 95% interval, and per-opponent W/D/L beside it
  (`metrics.versus`, for the head-to-head matrix). Same `EvaluationRecord`s as `evaluate.py`, so the game
  page's leaderboard components need nothing game-specific. Scores are relative to the field: re-run it
  whenever entrants change (it replaces the old records).
- `parallel.py` — `ProcessPoolEvaluator(inner, workers)`: scores a generation's genomes across processes
  (`evolve.fitness.evaluate_all` calls its `evaluate_many`). Fitness evaluators here are deterministic, so a parallel
  run reproduces a serial one; used by `checkers_neuro_run.py --workers N` and `checkers_distill_run.py`. Cost meters
  then report wall time, not summed CPU time of the workers.
- `checkers_distill_run.py` — search distillation for Checkers: evolve an evaluator to predict labelled positions
  (`--label-kind search|rollout|blend`; pools are cached under `run-data/distill/`), played `--depth` plies. A
  documented negative result (docs/design/0008): it never got near `Material 4-ply`.
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
