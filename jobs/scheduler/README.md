# scheduler

The scheduler workload (docs/design/0018, decision 2): **the only thing that turns queued jobs into running workers,
and the only thing that understands composite work.** The API records jobs and experiments; this runs them.

```bash
uv run python -m scheduler --workers 2      # Ctrl-C stops it, and (cleanly) the workers it started
uv run python -m scheduler --once           # one tick: recover, fan out, reap, claim -- for scripts and tests
```

Each tick (`scheduler.loop.Scheduler.tick`):

1. **Experiments** — every stored experiment's arm × seed cells with no completed run and no job in flight become
   trainer jobs (`parent` = the experiment). A cell whose arm continues another (`init_from`) waits until that run
   exists. A cell whose last job failed or was cancelled is retried only after the experiment is resubmitted
   (`POST /experiments` again), so a broken arm doesn't loop.
2. **Reap** — each worker it started: stopped if cancelled; `succeeded` with its result (the workload's `--result`
   file: `{"run_id": ...}`, a summary or a report) or `failed` with its exit code and log tail when it exits;
   heartbeat refreshed while it runs.
3. **Claim** — queued jobs, oldest first (an atomic claim in the job store: two schedulers never share a job), up to
   `--workers` at a time.

On start, `recover()` marks `running` jobs whose worker is gone `failed` — the crash rule `recorded_run()` has for
runs, one level up.

**Workers** (`scheduler.runner.LocalProcessRunner`): `python -m <trainer|evaluator|publisher> <spec> --result <file>`
in its own process group, output appended to `<data dir>/jobs/<job_id>.log` (what `GET /jobs/{id}/logs` serves).
Stopping never relies on Ctrl-C, which a console-less child often never receives: CTRL_BREAK_EVENT on Windows, SIGTERM
elsewhere — turned into KeyboardInterrupt by the workload (`jobcore.cli`), so its run is marked `failed` — then a kill
after `grace` seconds. A Kubernetes runner would implement the same `start`/`poll`/`stop`.

Not yet (stage 6b/6c): experiment report jobs, pipelines (the TD → TD-Leaf recipe), schedules.

Tests: `uv run pytest jobs/scheduler` — the decisions with a fake runner, and real worker processes (a job run end to
end; a cancel that stops the worker cleanly, its run marked `failed`).
