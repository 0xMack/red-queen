# trainer

The trainer workload (docs/design/0018): **one `TrainSpec` in, one recorded training run out.** A future container
image of its own; today a package you run locally.

```bash
uv run python -m trainer jobs/trainer/specs/snake-neat.yaml                       # NEAT on Snake, the defaults
uv run python -m trainer jobs/trainer/specs/checkers-neuroevolution.yaml \
    --set budget.generations=200 --set params.hidden=16 --set params.workers=8     # override any field
```

`--set` takes `key=value` with dotted keys (`seed`, `interface`, `budget.generations`, `params.neat.speciation`,
`tags.experiment`); the value is parsed as YAML. From Python: `trainer.train(spec)` takes a `TrainSpec` or a dict and
returns the run id. Runs land in the data directory (`telemetry.data_dir()`) like any other.

## How it fits together

- **The spec** (`jobcore.specs.TrainSpec`): `game`, `interface`, `algorithm`, `params`, `budget` (one unit, e.g.
  `{generations: 250}`), `seed`, `held_out_every`, `tags` (recorded in the run config, e.g. `experiment`/`arm`).
- **The params models** (`jobcore.algorithms`): one per algorithm *and game*: neuroevolution on Snake and on Checkers
  share the name, not their settings. Defaults are the original scripts' defaults. They live in `jobcore`, not here,
  so the backend and CLI can validate a spec without importing a trainer.
- **The adapters** (`trainer.algorithms`): `(spec, params, sink) -> run_id`, registered with `@adapter(name, *games)`.
  An adapter writes **exactly the run config the original script wrote** -- the frontend and the leaderboard jobs read
  those keys -- and `jobs/tests/test_golden.py` holds every adapter to its script's golden run: config, curve and
  champion bytes.
- `trainer.snake` / `trainer.checkers`: what each game's algorithms share (rollouts, the telemetry callback and
  held-out monitor; opponent pools, hall of fame, material seeding, the margin scorer).
- `specs/`: one example per (algorithm, game), at the defaults (a test checks they resolve and set nothing the model
  doesn't default).

## Algorithms

| Spec | What it trains |
|---|---|
| `gp.yaml` | Linear GP on the fixed symbolic-regression benchmark (docs/design/0003 phase 1) -- the reference example of wiring an algorithm to telemetry. Its champion artifact is a prototype `repr` (function addresses included, so the bytes differ run to run). |
| `snake-neuroevolution.yaml` | A policy network under a named interface (docs/design/0007), lexicase over the training games, every episode played by the Rust core. `params.seeds`: `fixed:5` (the original: the champion memorizes its 5 games -- its unseen-game score peaks around generation 20-25, then falls) or `resample:N` (fresh games every generation). Held-out score on `arena.snake.MONITOR_SEEDS` every `held_out_every`. |
| `snake-neat.yaml` | NEAT (docs/design/0008): speciation + fitness sharing, `extras` carry species count and champion size. `params.neat` overrides `NeatConfig` (`speciation: false` is the ablation). ~1.5 min for 250 generations. |
| `checkers-neuroevolution.yaml` | A 32 -> H -> 1 position evaluator searched `depth` plies, fitness = match outcomes against fixed opponents (one case per opponent per seat, lexicase), optional hall of fame (`hall`), material seeding (`seed_material`), draws scored by material (`margin`), fresh opponent games every generation (`resample` -- fixed games get memorized), `workers` for parallel scoring (same result). |
| `checkers-neat.yaml` | The NEAT counterpart: a graph growing from a linear start, gentler weight mutation than NEAT's default. |
| `checkers-distill.yaml` | Search distillation: fit the evaluator to a deeper material search's values on a cached position pool (`data/distill/`). A documented negative result (docs/design/0008). The pool depends on `workers`. |
| `bandit-evolve.yaml` | Evolves epsilon-greedy's four settings for the bandit (docs/design/0011) on fresh training games; `jobs/evaluate_bandit.py` ranks every completed run's champion. |

Not yet ported (stage 3b/3c of docs/design/0018): RL (`jobs/rl_run.py`), Checkers self-play and PBT
(`jobs/checkers_selfplay_run.py`, `checkers_pbt_run.py`), TinyLM (`jobs/tinylm_run.py`).

Tests: `uv run pytest jobs/trainer` (and the golden runs, `uv run pytest jobs/tests/test_golden.py`).
