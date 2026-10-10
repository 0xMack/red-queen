# 0018 — Workloads, a job API, and the `redqueen` CLI

Status: **Accepted; stage 1 built** (2026-10-10). Agreed with Mack in an interview on 2026-10-10 (the decisions below
record the answers). Stage 1 (foundations: `telemetry.data_dir()`/`open_stores()`, `libs/arena`, the deletions and
moves) is built; stages 2-8 are not.
Relates to: [0001](0001-fast-cpp-gp-pybind11.md) (algorithm libs never import `telemetry`),
[0002](0002-realtime-visualization-architecture.md) (log-then-serve; distributed compute),
[0005](0005-frontend-and-api-contracts.md) (one API module, reuse pydantic models, extract a module when something forces
it), [0007](0007-representations-leaderboards-and-tradeoffs.md) (protocols, interfaces),
[0008](0008-neat-and-tracked-comparisons.md) / [0010](0010-reinforcement-learning.md) /
[0014](0014-two-player-training-regimes.md) (the experiments being converted).

## Context

`jobs/` was meant to be a parent of long-running jobs and workers. It became 36 loose scripts (6,240 lines) and 21
separate argparse entry points:

- **Not a package.** No `pyproject.toml`. It imports only because `uv run python jobs/x.py` puts the script's directory on
  `sys.path`, and `jobs/tests/conftest.py` repeats that trick. Scripts import scripts: `checkers_selfplay_experiment`
  imports eight siblings, two of them other games' experiments, and private helpers cross files
  (`from snake_experiment import _stats, parse_seeds`).
- **Domain code lives in scripts.** `evaluate.py` owns `PROTOCOL`, `HELD_OUT_SEEDS` and `MONITOR_SEEDS`, so
  `snake_neuro_run`, `rl_run` and `publish_models` import the leaderboard *script* to get seed ranges. `versus_stats.py`
  (which has a TypeScript twin), `costs.py` and `seeding.py` are libraries.
- **One kind of thing, written twelve times.** Each `*_run.py` is "train algorithm X on game Y under interface Z with
  params P, record it". Each `*_experiment.py` is "arms × seeds of those, then a report". Each `evaluate*.py` is "score
  entrants under protocol Q". Flags, `run_id=` printing, run-id prefix matching (`checkers_sprt`) and `run`/`report`
  subcommands are re-implemented each time. The arms are Python dicts. Recipes that need several steps (train, then
  evaluate, publish and re-rank the versus leaderboard; or a TD phase followed by a TD-Leaf fine-tune) live in prose.
- **The storage layer leaks.** Two pieces of code open the stores (`jobs/run_context.TelemetryStores` and
  `apis/backend/dependencies.py`). Two places define the data directory with different defaults: the backend resolves
  `jobs/run-data` relative to the current directory, the jobs relative to their own file, so everything must be launched
  from the repo root. `backfill_interfaces.py` writes `runs.db` with raw `sqlite3` because `RunRegistry` can't update a
  config. Run data lives inside the source tree.
- **The API is a read-only window.** It has no resource for jobs, experiments, evaluations or publishing.
  "Experiment" exists only as `config.experiment`, re-parsed by every script. An agent, or the CLI, couldn't start or
  follow work through it.

## Target state

```
libs/
  telemetry/   + data_dir(), open_stores() [built]; RunRegistry.update_config, JobStore (jobs + experiments)
  arena/       NEW. Measurement as a domain: protocols (snake.score.v2, checkers.versus.v2, bandit.skill.v1,
               checkers SPRT), their seed sets (HELD_OUT, MONITOR, training pools), versus_stats, the training-cost
               meter, seed strategies. No I/O, no telemetry.
  cli/         NEW. `redqueen` (alias `rq`, decision 8): Typer, an HTTP client over the backend, `--json`
               everywhere, later `redqueen mcp serve`.
jobs/                                   one uv package per workload; each a future container image
  core/        `jobcore`: spec models (TrainSpec, EvalSpec, PublishSpec, ExperimentSpec, Job), the schema registry,
               the RunSink protocol + LocalSink/HttpSink, run lifecycle, pause/resume control, ProcessPool evaluator
  trainer/     `trainer`: runs ONE TrainSpec. algorithms/{gp,neuroevolution,neat,bandit_evolve,distill,rl,selfplay,
               alphazero,pbt,tinylm}.py, each an adapter with a pydantic params model
  evaluator/   `evaluator`: runs ONE EvalSpec. protocols/{snake_score,checkers_versus,bandit_skill,checkers_sprt,
               round_robin}.py. Writes EvaluationRecords through the sink
  publisher/   `publisher`: runs ONE PublishSpec (export champions/TinyLM to the model store, catalog)
  scheduler/   `scheduler`: a long-running daemon. Claims queued jobs and launches the right workload through a
               Runner (LocalProcessRunner now, KubernetesRunner later), fans experiments out into trainer jobs and
               enqueues their report, resolves pipelines (stage N's init_from = stage N-1's run), runs cron schedules
  scripts/     jobs-level one-offs only
experiments/   checked-in ExperimentSpec YAML: snake-neat.yaml, dqn-ladder.yaml, selfplay-v1..v8.yaml, ...
apis/backend   services/ (routers become thin); write endpoints for HttpSink; /jobs, /experiments, /specs
data/          the default data directory (gitignored), replacing jobs/run-data
```

Dependency direction (arrows = imports): `trainer|evaluator|publisher|scheduler → jobcore → arena, telemetry → (nothing)`;
`trainer → evolve, rl, games, tinylm`; `backend → jobcore (specs only), telemetry, arena, modelpack`;
`cli → jobcore (specs only), httpx`. Algorithm libs still know nothing about any of it (0001). `jobcore` stays
pydantic + telemetry only, so the backend and CLI can validate specs without pulling in a trainer.

## Decisions

### 1. Jobs are a few generic workloads, chosen by payload, not 36 scripts

A workload is a deployable unit: one image, one entrypoint (`python -m trainer <job-id | spec.yaml>`), one payload
type. All twelve training scripts become one trainer with algorithm adapters. All five evaluation scripts become one
evaluator with protocol plugins. Different work means different payloads to the same endpoint, not a new script:

```yaml
# TrainSpec
kind: train
game: checkers
interface: checkers/board32.v1+evaluate1ply.v1
algorithm: selfplay.td_lambda        # discriminator -> SelfPlayParams
params: {hidden: 64, hidden_layers: 2, pool_size: 10, search_depth: 1}
budget: {games: 1_000_000}            # or generations / env_steps / iterations: the algorithm declares which
seed: 3
init_from: {run: 672890ba}            # or {experiment: selfplay-v2, arm: pool-2x64-1m, seed: 3}
monitor: {held_out_every: 10}
tags: {experiment: selfplay-v6, arm: l-3x64-1m}
```

`params` is a discriminated union keyed by `algorithm`: each adapter owns its pydantic model, registered in `jobcore`.
`GET /specs/schemas` exposes the JSON Schema of every kind and algorithm, which is what CLI help and MCP tool schemas are
generated from.

**Compatibility constraint:** the trainer writes exactly the run `config` keys the frontend and evaluator read today
(`game`, `interface`, `representation`, `paradigm`, `experiment`, `arm`, `rng_seed`, `training_seeds`, ...). Existing
runs stay attributable and evaluable unchanged. **Acceptance test per adapter:** a short golden run from the old script's
defaults and the equivalent spec produce byte-identical champions. Every learner here is deterministic per seed, so this
is a real equality test, not a tolerance.

### 2. Four workloads: trainer, evaluator, publisher, and a scheduler

The API stays thin: it validates, records and serves. It doesn't supervise processes. The **scheduler** is the only
thing that turns a queued `Job` into a running process, and the only thing that understands composite work:

- **Experiments:** an `ExperimentSpec` (arms × seeds, each arm a partial TrainSpec plus a `baseline` arm and an optional
  `init_from` that may name another experiment's arm, as `selfplay-v5`/`v6` already do) expands into trainer jobs. When
  they all finish, the scheduler enqueues one evaluator job of kind `experiment-report` (held-out games, or `h2h`).
  Resumable for free: arms × seeds already completed are skipped, as the scripts do today.
- **Pipelines:** an ordered list of specs where a stage can reference the previous stage's run. `checkers_recipe_run`
  (TD, then TD-Leaf) becomes a two-stage pipeline file.
- **Schedules:** cron entries submit specs, e.g. re-rank `checkers.versus.v2` nightly or whenever a Checkers run
  completes.
- **PBT stays one trainer job** (`algorithm: pbt`): its members exploit and explore every round in-process, and making
  each round a scheduler fan-out is a separate design. Revisit if PBT populations need more than one machine.

Runner: `LocalProcessRunner` starts `uv run --package <workload> python -m <workload> <job-id>` in a new process group,
writes its log under `data/jobs/<id>.log`, and records the pid plus a heartbeat. On restart, the scheduler marks jobs
whose process is gone or whose heartbeat has stalled as `failed`. This is the crash-safety rule `recorded_run()` already
has, applied one level up. A `KubernetesRunner` (a Job/CronJob per workload image, on the homelab MicroK8s) can be added
behind the same protocol later; nothing in this plan builds it.

### 3. A pluggable run sink: local by default, HTTP when the API is the only writer

`RunSink` (in `jobcore`) is everything a workload writes: create run, record generation, put artifact, set summary/status,
update config, read status (for pause/resume), put/prune evaluation records.

- `LocalSink` writes through `telemetry.open_stores(data_dir())`: log-then-serve, exactly 0002's decision. A run never
  depends on the server being up. It's the default and what local development uses.
- `HttpSink` writes through new backend endpoints (below). It's for workers that can't share the data directory, like a
  cluster pod. It batches metrics per generation.

Selected per job (`sink:` in the spec, or `REDQUEEN_SINK=local|http://host:8000`). `recorded_run()` and the control
callback move onto the sink unchanged in behaviour. The publisher writes the local model store only, until the store's
host (R2 or the HF Hub, still undecided per 0009) is chosen. A remote publisher is out of scope.

### 4. `libs/arena`: measurement is domain code, not a job

Protocols (seed sets, caps, board, scoring rule, version), `MONITOR_SEEDS`, training-seed strategies, `versus_stats`
(Elo, pentanomial counts, Bradley–Terry, SPRT), and the cost meter plus hardware fingerprint. Training imports it for
monitoring seeds, the evaluator for protocols, the backend to describe protocols. `checkers_training` (opponent pools,
margin scorer, material seeding) is training-only and moves into the trainer's Checkers adapters. The "never edit a
protocol in place, bump the version" rule moves with the protocols.

### 5. The API owns the schema; the CLI follows its slices and orchestrates

Routers become thin over `backend/services/` (runs, evaluations, jobs, experiments, models). New resources:

| Resource | Endpoints | For |
|---|---|---|
| runs (write) | `POST /runs`, `PATCH /runs/{id}` (status, summary, config merge), `POST /runs/{id}/metrics`, `PUT /runs/{id}/artifacts/{ref}` | HttpSink. Config merge replaces `backfill_interfaces`' raw sqlite |
| evaluations | `PUT /games/{game}/evaluations`, `POST .../prune` | evaluator over HttpSink |
| jobs | `POST /jobs {kind, spec}` (422 on a bad spec), `GET /jobs?kind&status&parent`, `GET /jobs/{id}`, `GET /jobs/{id}/logs` (SSE tail, same polling pattern as `/metrics/stream`), `POST /jobs/{id}/cancel` | CLI, MCP, scheduler |
| experiments | `POST /experiments` (an ExperimentSpec), `GET /experiments`, `GET /experiments/{name}` (arm × seed matrix: job, run, status), `GET /experiments/{name}/report` | CLI, Learn citations |
| specs | `GET /specs/schemas` | CLI help, MCP tool schemas |

The CLI is an **HTTP client only** (`REDQUEEN_API_URL`, default `http://localhost:8000`). It reuses the same pydantic
models (`telemetry`, `jobcore`) rather than generating a client: the 0005 reuse rule, with the OpenAPI snapshot test as
the drift guard. Command groups follow the schema slices, and verbs wrap multi-call work:

```
redqueen runs        list | show | history | tail | pause | resume | step | artifact | brain
redqueen leaderboard show GAME [--protocol] | interfaces GAME
redqueen models      catalog GAME | package RUN REF
redqueen jobs        submit SPEC.yaml | list | show | logs [-f] | cancel
redqueen train       ALGORITHM --game --interface --param k=v ... [--wait]    # builds a TrainSpec, POST /jobs
redqueen evaluate    GAME [--protocol] [--wait]
redqueen h2h         A B [--full]                                            # checkers SPRT, run-id prefixes
redqueen publish     GAME [--entrant ...]
redqueen experiments submit FILE.yaml | list | status NAME | report NAME [--h2h]
redqueen pipeline    submit FILE.yaml
redqueen export      ballot | pbt-history RUN | rl-curves NAME | rl-recording RUN   # writes apps/frontend/app/data
redqueen dev         up | build-wasm [games|rl] | openapi | bench-rl | screenshots RUN
```

Every command takes `--json` (machine output, stable shapes), and every command that starts work takes `--wait`
(follow logs until it ends, exit code = job outcome). Run ids resolve by unique prefix everywhere. `dev` and
`export ballot` are the only local-only commands (they build or generate files in the repo).

### 6. MCP last, over the CLI's own command layer

`redqueen mcp serve` exposes the same commands as tools: typed parameters from the Typer signatures, JSON results, and
long work as `submit` → `status`/`logs` → `cancel`, never one blocking call. Building the CLI with `--json` and job
handles from the start is what makes this a thin final step.

### 7. Housekeeping agreed

- Delete `bandit_arm_sweep.py` (a pre-Rust sketch) and `backfill_interfaces.py` (a migration that has already run); git
  history keeps both.
- The `export_*` scripts become `redqueen export ...`.
- `libs/{games,rl}/build-wasm.py` → `libs/{games,rl}/scripts/build_wasm.py`, also available as `redqueen dev build-wasm`.
  `rl_benchmark.py` → `libs/rl/scripts/benchmark.py` (`redqueen dev bench-rl`).
- `REDQUEEN_DATA_DIR` (default `<repo>/data/`), defined once in `telemetry.data_dir()`. `REDQUEEN_RUN_DATA_DIR` is kept
  as an alias. Existing data moves once: `mv jobs/run-data data`.
- The workspace adds `"jobs/*"` to `members` (`libs/cli` joins through `libs/*`).

### 8. The command is `redqueen`; `rq` is a convenience alias that nothing depends on

Both are console scripts for the same entry point, but `rq` is a common name: RQ (Redis Queue) installs an `rq`
command, and so does a Rust record-query tool. Inside the project (`uv run rq`, or an activated venv) the venv's own
`Scripts/` is searched first, so ours wins; outside it, PATH order decides silently, and a shadowing tool would just
misread our arguments -- there is no graceful fallback to rely on. So:

- docs, AGENTS.md, CI, the MCP server and every script call `redqueen`, never `rq`;
- `redqueen doctor` reports whether `rq` on PATH resolves to us, and names the other tool when it doesn't;
- a test fails if any workspace dependency also ships an `rq` script -- the one clash that would otherwise overwrite
  ours inside the venv without a warning (and RQ is a plausible pick for a job queue).

## Every current script, and where it goes

| Script | Destination |
|---|---|
| `baseline_gp_run` | trainer `gp` |
| `snake_neuro_run`, `checkers_neuro_run` | trainer `neuroevolution` (game-generic; Checkers opponents/depth are params) |
| `snake_neat_run`, `checkers_neat_run` | trainer `neat` |
| `checkers_distill_run` | trainer `distill` (a negative result; see open questions) |
| `bandit_evolve_run` | trainer `bandit_evolve` |
| `rl_run` | trainer `rl.{q_learning,sarsa,dqn,reinforce,a2c,ppo}` |
| `checkers_selfplay_run` | trainer `selfplay.td_lambda`, `alphazero` |
| `checkers_pbt_run` | trainer `pbt` |
| `tinylm_run` | trainer `tinylm` |
| `checkers_recipe_run` | `experiments/pipelines/checkers-recipe.yaml` |
| `snake_experiment`, `rl_experiment`, `checkers_selfplay_experiment` | scheduler fan-out + `experiments/*.yaml`; reports → evaluator `experiment-report` |
| `evaluate`, `evaluate_versus`, `evaluate_bandit` | evaluator protocols (definitions in arena) |
| `checkers_sprt`, `checkers_round_robin` | evaluator `checkers_sprt`, `round_robin` |
| `publish_models`, `scale_test_package` | publisher (`scale-test` a spec flag) |
| `export_ballot`, `export_pbt_history`, `export_rl_curves`, `export_rl_recording` | `redqueen export ...` |
| `rl_benchmark` | `libs/rl/scripts/benchmark.py` |
| `versus_stats`, `costs`, `seeding`, protocol constants | `libs/arena` |
| `run_context`, `control`, `parallel` | `jobcore` (onto the sink) |
| `checkers_training` | trainer Checkers adapters |
| `bandit_arm_sweep`, `backfill_interfaces` | deleted |

## Delivery: staged PRs, each green in CI with the docs updated

1. **Foundations** (no behaviour change) -- *built*: `telemetry.data_dir()/open_stores()`, `data/` default, `libs/arena`
   extracted (scripts re-pointed at it), deletions, build-wasm and the RL benchmark moved. The old scripts still run.
   (`RunRegistry.update_config` moved to stage 5, where its first caller, `PATCH /runs/{id}`, lands.)
2. **`jobcore`:** spec models + registry, `RunSink` + `LocalSink`, run lifecycle and control on the sink. Golden-champion
   fixtures captured from the *old* scripts here, before anything is ported.
3. **trainer** in three PRs: (a) evolution family: gp, neuroevolution, neat, bandit_evolve, distill; (b) rl, selfplay,
   alphazero, pbt; (c) tinylm. Each deletes the scripts it replaces, once its golden tests pass. Interim invocation:
   `uv run --package trainer python -m trainer spec.yaml`.
4. **evaluator + publisher**, with the same golden approach (identical `EvaluationRecord`s on a small field).
5. **backend:** services layer, write endpoints + `HttpSink` (with `RunRegistry.update_config`), `JobStore`, `/jobs`,
   `/experiments`, `/specs`. OpenAPI snapshot and frontend types regenerated.
6. **scheduler:** queue claiming, `LocalProcessRunner`, experiment fan-out and reports, pipelines, cron. Every existing
   experiment converted to YAML under its existing name, and the `*_experiment.py` scripts deleted.
7. **`redqueen` CLI:** every group above, `export` and `dev` included. AGENTS.md, the READMEs and the design docs'
   "run with" lines rewritten to CLI commands. `jobs/README.md` becomes a workload index.
8. **MCP:** `redqueen mcp serve`.

## Risks and open questions

- **Windows process control.** Cancelling a worker must not depend on `CTRL_C_EVENT` delivery (the uvicorn `--reload`
  lesson in AGENTS.md). Use `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT`, then terminate after a grace period, and
  have the run marked `failed`/`cancelled` by the sink's lifecycle, not the killer.
- **Generalising neuroevolution across games** (the Snake and Checkers runs differ in fitness evaluators, opponents and
  hall of fame). If a shared adapter needs too many game switches, use two adapters behind one algorithm name. The
  golden tests decide, not taste.
- **Settled in review (2026-10-10):** `distill` is ported, not deleted (0008 cites it); workload import names stay
  bare (`jobcore`, `trainer`, ...), matching `games`/`rl`; cancellation uses the process-group approach above.
- **Auth:** none, per 0005 -- deferred. A job API that spawns processes is fine on localhost, but becomes a real
  question the moment the backend is reachable from the homelab network; nothing here exposes it beyond localhost.
