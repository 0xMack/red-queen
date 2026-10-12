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
  `distill`, `bandit_evolve`, `rl.<algorithm>`, `td_lambda`, `alphazero`, `pbt`, `tinylm` -- they replaced `baseline_gp_run.py`,
  `snake_neuro_run.py`, `snake_neat_run.py`, `checkers_neuro_run.py`, `checkers_neat_run.py`, `checkers_distill_run.py`,
  `bandit_evolve_run.py`, `rl_run.py`, `checkers_selfplay_run.py`, `checkers_pbt_run.py` and `tinylm_run.py`,
  reproducing each one's
  golden runs exactly. See `trainer/README.md` for what each algorithm does and records.
- **`evaluator/` — the evaluator workload**: one `EvalSpec` (a leaderboard protocol or a report) in, its result out.
  `uv run python -m evaluator evaluator/specs/<protocol>.yaml [--set key=value ...]`. Replaced `evaluate.py`,
  `evaluate_versus.py`, `evaluate_bandit.py`, `checkers_sprt.py` and `checkers_round_robin.py`; see
  `evaluator/README.md`.
- **`scheduler/` — the scheduler**: `uv run python -m scheduler`; claims queued jobs (`POST /jobs`) and runs them as
  worker processes, and fans stored experiments out into jobs. See `scheduler/README.md`.
- **`publisher/` — the publisher workload**: one `PublishSpec` (target `snake`, `tinylm` or `scale-test`) in, packages in
  the model store out. Replaced `publish_models.py` and `scale_test_package.py`; see `publisher/README.md`.
- `snake_experiment.py` — a tracked *comparison*, each arm a partial `TrainSpec` run by the trainer: arms
  (`neuro-lexicase`, `neuro-tournament`, `neat`,
  `neat-no-speciation`) × rng seeds, every run recorded to telemetry and tagged `config.experiment`/
  `arm`/`rng_seed`, resumable, arms parallelizable as separate processes. `report --name NAME`
  scores each run's *final* champion on the 200 leaderboard games and writes per-arm aggregates to
  `data/experiments/NAME.json` (gitignored — the durable record of a result is
  docs/design/0008 and the Learn chapter). Experiment-tagged runs are deliberately kept off the
  leaderboard (the Snake evaluation skips them) so 20 seeds don't bury every other entrant.
- `export_ballot.py` — writes the ballot (every 3-ply opening, its board and verdict) as
  `apps/frontend/app/data/checkersBallot.ts` for the Learn chapter.
- **Reinforcement learning** (`libs/rl`, docs/design/0010): the trainer's `rl.<algorithm>` trains one `rl.Trainer`
  algorithm, an *iteration* (a budget of env steps) per recorded generation; `rl_experiment.py` is its tracked comparison (arms
  budgeted in env steps, final policy on the 200 held-out games, a paired permutation test against each arm's baseline);
  `export_rl_recording.py` / `export_rl_curves.py`
  write real runs as the Learn chapters' recorded fallbacks.
- **Checkers self-play** (docs/design/0010 Phase 4, 0014-0017): the trainer's `td_lambda` trains a position evaluator
  by TD(λ) or TD-Leaf(λ) (`params.agent.search_depth: N`), `alphazero` by AlphaZero-style search targets, each recorded
  as an ordinary Checkers run; `checkers_selfplay_experiment.py` runs and compares the tracked arms (`selfplay-v1` ...
  `v8`); `checkers_recipe_run.py` is the best recipe (plain TD, then a TD-Leaf fine-tune) as one command, for leaderboard
  entrants (a pipeline of two trainer runs); the trainer's `pbt` is population-based training over self-play learners,
  and `export_pbt_history.py` writes
  one run's population history for the Learn chapter.
- **Bandits** (docs/design/0011): the evaluator's `bandit.skill.v1` is the bandit leaderboard;
  the trainer's `bandit_evolve` evolves ε-greedy's settings as an ordinary run.
- **Model packages and TinyLM** (docs/design/0004, 0009): the publisher exports leaderboard champions and TinyLM
  checkpoints to the local model store and catalog; the trainer's `tinylm` trains TinyLM.
- **Shared plumbing is the `jobcore` package (`jobs/core/`)**: `recorded_run()` (create the run, mark it `completed`,
  or `failed` if the job dies), the pause/resume control callback, `open_sink()` (where a job writes; local by
  default), `ProcessPoolEvaluator` (`--workers N`), and the job payload specs. See `jobs/core/README.md`.
- **Golden runs** (`tests/golden_cases.py`, `tests/golden/champions.<platform>.json`): a small deterministic run of every training
  job, reduced to its config, curve and champion bytes, recorded before any job was ported to the trainer workload.
  `tests/test_golden.py` re-runs them (~9 s); `tests/golden_eval.py` / `test_golden_eval.py` do the same for every
  evaluation protocol, report and publish target, on a small fixed field (~45 s). Re-record only for a deliberate behaviour change:
  `uv run python jobs/tests/golden_cases.py --record <case>`. Exact per OS, not across (Windows' and Linux's libm
  differ in the last bit); the Linux fixture is recorded by CI, which uploads its digests as an artifact on failure.

`jobs/` isn't a `uv` workspace package (no `pyproject.toml`) — scripts here import already-installed
workspace packages, and `uv run python jobs/<script>.py` puts the script's own directory on
`sys.path` automatically, which is how e.g. `rl_experiment.py` imports `snake_experiment.py` as a plain
sibling module. `jobs/tests/conftest.py` does the same thing explicitly for `uv run pytest
jobs/tests`.
