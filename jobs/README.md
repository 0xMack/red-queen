# jobs

The workloads (docs/design/0018). Each is a workspace package that takes one spec in and produces a result. They are
the only place algorithm libs get wired to `telemetry`; the libs never import it themselves (docs/design/0001).

Drive them through the API with the `redqueen` CLI (`libs/cli`):

```bash
uv run redqueen dev up                         # the backend, the scheduler and the frontend; Ctrl-C stops all three
uv run redqueen train td_lambda --game checkers --budget iterations=200 -p agent.hidden=64 --wait
uv run redqueen evaluate checkers --wait       # re-rank a leaderboard
uv run redqueen experiments submit experiments/selfplay-v8.yaml
```

Any workload also runs directly on a spec file, with no API and no scheduler:
`uv run python -m trainer|evaluator|publisher SPEC [--set key=value ...]`.

| Workload | Takes | Does | CLI |
|---|---|---|---|
| [`trainer/`](trainer/README.md) | a `TrainSpec` | trains one recorded run: `gp`, `neuroevolution`, `neat`, `distill`, `bandit_evolve`, `rl.<algorithm>`, `td_lambda`, `alphazero`, `pbt`, `tinylm` | `redqueen train` |
| [`evaluator/`](evaluator/README.md) | an `EvalSpec` | a leaderboard protocol (`snake.score.v2`, `checkers.versus.v2`, `bandit.skill.v1`) or a report (`checkers.sprt`, `checkers.static-round-robin`, `experiment.*`) | `redqueen evaluate`, `redqueen h2h` |
| [`publisher/`](publisher/README.md) | a `PublishSpec` | model packages (`snake`, `tinylm`, `scale-test`) and the frontend's data files (`frontend.*`) | `redqueen publish`, `redqueen export` |
| [`scheduler/`](scheduler/README.md) | the job store | runs queued jobs as worker processes, and fans out experiments, pipelines and schedules | `redqueen jobs`, `experiments`, `pipeline`, `schedules` |
| [`core/`](core/README.md) (`jobcore`) | — | what they share: the sink (local or HTTP), the run lifecycle, pause/resume, `ProcessPoolEvaluator`, every spec and registry | — |

Checked-in specs live in [`experiments/`](../experiments): tracked comparisons (`*.yaml`), pipelines
(`pipelines/`, e.g. the TD → TD-Leaf recipe for leaderboard entrants) and schedules (`schedules/`). Each workload
also ships one example spec per algorithm, protocol or target under its own `specs/` directory.

Experiment-tagged runs stay off the leaderboards, so that 20 seeds don't bury every other entrant. A comparison is
reported by its experiment's report protocol, which writes `data/experiments/NAME.json`. That file is gitignored: the
durable record of a result is the design doc and the Learn chapter.

**Golden runs** (`tests/golden_cases.py`, `tests/golden/champions.<platform>.json`) are a small deterministic run of
every training job, reduced to its config, curve and champion bytes. They were recorded from the original scripts
before any job was ported. `tests/test_golden.py` re-runs them (~9 s). `tests/golden_eval.py` /
`test_golden_eval.py` do the same for every evaluation protocol, report and publish target, on a small fixed field
(~45 s). Re-record only for a deliberate behaviour change:
`uv run python jobs/tests/golden_cases.py --record <case>`. The fixtures are exact per OS, not across OSes, because
Windows' and Linux's libm differ in the last bit. CI records the Linux fixture: on a failure it uploads its digests as
an artifact.
