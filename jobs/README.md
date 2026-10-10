# jobs

> **Being restructured (docs/design/0018).** These scripts are becoming workloads — a trainer, an evaluator, a
> publisher and a scheduler — behind a job API and the `redqueen` CLI. Shared measurement code (protocols, seed sets,
> `versus_stats`, the cost meter, seed strategies) has already moved to `libs/arena`.

Short- or long-running jobs, tasks, and workers — e.g. training runs, batch evaluations, or
background simulation workers. This is where algorithm libs (which never import `telemetry`
themselves — see docs/design/0001) get wired to it for a real run.

## Contents

- **`trainer/` — the trainer workload** (`trainer` package): one `TrainSpec` in, one recorded run out.
  `uv run python -m trainer trainer/specs/<algorithm>.yaml [--set key=value ...]` (paths from `jobs/`), one example
  spec per algorithm at the original scripts' defaults. Built: `gp`, `neuroevolution` and `neat` (Snake and Checkers),
  `distill`, `bandit_evolve` -- they replaced `baseline_gp_run.py`, `snake_neuro_run.py`, `snake_neat_run.py`,
  `checkers_neuro_run.py`, `checkers_neat_run.py`, `checkers_distill_run.py` and `bandit_evolve_run.py`, reproducing
  each one's golden runs exactly. See `trainer/README.md` for what each algorithm does and records.
- `snake_experiment.py` — a tracked *comparison*, each arm a partial `TrainSpec` run by the trainer: arms
  (`neuro-lexicase`, `neuro-tournament`, `neat`,
  `neat-no-speciation`) × rng seeds, every run recorded to telemetry and tagged `config.experiment`/
  `arm`/`rng_seed`, resumable, arms parallelizable as separate processes. `report --name NAME`
  scores each run's *final* champion on the 200 leaderboard games and writes per-arm aggregates to
  `data/experiments/NAME.json` (gitignored — the durable record of a result is
  docs/design/0008 and the Learn chapter). Experiment-tagged runs are deliberately kept off the
  leaderboard (`evaluate.py` skips them) so 20 seeds don't bury every other entrant.
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
- `evaluate_versus.py` — the two-player leaderboard (docs/design/0007's "Versus", 0013): protocol
  `checkers.versus.v2`, a round robin over every finished checkers champion plus the fixed baselines, 12 ballot
  openings (`games.checkers_openings`) per pairing, each a **game pair** (both seats). An entrant's score
  (`quality.mean`) is an **Elo rating**: Bradley–Terry over every game, Random = 0, bootstrap 95% interval
  (`arena.versus_stats`); points per game, pentanomial pair counts and per-opponent W/D/L are in `metrics.versus`. Same
  `EvaluationRecord`s as `evaluate.py`, so the game page's leaderboard components need nothing game-specific. Ratings
  are relative to the field: re-run it whenever entrants change (it replaces the old records).
- `checkers_sprt.py` — is player A stronger than B? Game pairs over the ballot (fixed shuffled order) until the SPRT
  decides; a player is a fixed strategy (`material-6`) or a run id prefix (`672890ba@4`). `checkers_selfplay_experiment.py
  h2h` runs the same test for every arm against its baseline arm, seed for seed (`--full`: the whole ballot).
- `export_ballot.py` — writes the ballot (every 3-ply opening, its board and verdict) as
  `apps/frontend/app/data/checkersBallot.ts` for the Learn chapter.
- **Reinforcement learning** (`libs/rl`, docs/design/0010): `rl_run.py` trains one `rl.Trainer` algorithm, an
  *iteration* (a budget of env steps) per recorded generation; `rl_experiment.py` is its tracked comparison (arms
  budgeted in env steps, final policy on the 200 held-out games, a paired permutation test against each arm's baseline);
  `export_rl_recording.py` / `export_rl_curves.py`
  write real runs as the Learn chapters' recorded fallbacks.
- **Checkers self-play** (docs/design/0010 Phase 4, 0014-0017): `checkers_selfplay_run.py` trains a position evaluator by
  TD(λ), TD-Leaf(λ) (`--param search_depth=N`) or AlphaZero-style search targets (`--algorithm alphazero`) and records it
  as an ordinary Checkers run; `checkers_selfplay_experiment.py` runs and compares the tracked arms (`selfplay-v1` ...
  `v8`); `checkers_recipe_run.py` is the best recipe (plain TD, then a TD-Leaf fine-tune) as one command, for leaderboard
  entrants; `checkers_pbt_run.py` is population-based training over self-play learners, and `export_pbt_history.py` writes
  one run's population history for the Learn chapter.
- **Bandits** (docs/design/0011): `evaluate_bandit.py` is the bandit leaderboard (protocol `bandit.skill.v1`);
  the trainer's `bandit_evolve` evolves ε-greedy's settings as an ordinary run.
- **Model packages and TinyLM** (docs/design/0004, 0009): `publish_models.py` exports leaderboard champions and TinyLM
  checkpoints to the local model store and catalog, measuring each variant's agreement; `tinylm_run.py` trains TinyLM;
  `scale_test_package.py` publishes a large random-weight model to exercise the big-model path.
- **Shared plumbing is the `jobcore` package (`jobs/core/`)**: `recorded_run()` (create the run, mark it `completed`,
  or `failed` if the job dies), the pause/resume control callback, `open_sink()` (where a job writes; local by
  default), `ProcessPoolEvaluator` (`--workers N`), and the job payload specs. See `jobs/core/README.md`.
- **Golden runs** (`tests/golden_cases.py`, `tests/golden/champions.<platform>.json`): a small deterministic run of every training
  job, reduced to its config, curve and champion bytes, recorded before any job was ported to the trainer workload.
  `tests/test_golden.py` re-runs them (~9 s). Re-record only for a deliberate behaviour change:
  `uv run python jobs/tests/golden_cases.py --record <case>`. Exact per OS, not across (Windows' and Linux's libm
  differ in the last bit); the Linux fixture is recorded by CI, which uploads its digests as an artifact on failure.

`jobs/` isn't a `uv` workspace package (no `pyproject.toml`) — scripts here import already-installed
workspace packages, and `uv run python jobs/<script>.py` puts the script's own directory on
`sys.path` automatically, which is how e.g. `checkers_pbt_run.py` imports `checkers_selfplay_run.py` as
a plain sibling module. `jobs/tests/conftest.py` does the same thing explicitly for `uv run pytest
jobs/tests`.
