"""The scheduler (docs/design/0018, decision 2): the only thing that turns queued jobs into running workers, and the only
thing that understands composite work.

One `tick()`:
1. **experiments** -- every stored experiment's arm x seed cells that have no completed run and no job in flight
   become trainer jobs (`parent` = the experiment). A cell whose arm continues another (`init_from`) waits until that
   run exists. A cell whose last job failed or was cancelled is retried only once the experiment has been resubmitted
   since (so a broken arm doesn't loop forever);
2. **reap** -- each worker it started: cancelled if asked (the runner stops it), `succeeded` with its result or
   `failed` with its log's tail once it exits, its heartbeat refreshed while it runs;
3. **pipelines** -- each pipeline job's next stage is submitted (`parent` = the pipeline's job id) once the last
   succeeded, its `init_from: {stage}` filled in with that stage's run; the pipeline fails with the first stage that
   does, and succeeds with every stage's result. A pipeline has no worker of its own;
4. **claim** -- queued jobs, oldest first, up to `max_workers` running at once.

`recover()` runs once at start: a `running` job whose worker is gone (this scheduler restarted, or the machine did) is
marked `failed` -- the crash rule `recorded_run()` has for runs, one level up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from jobcore.specs import ExperimentSpec, PipelineSpec
from telemetry import Job, JobStore, RunRegistry

from scheduler.runner import Handle, LocalProcessRunner, Runner, log_tail, pid_alive, read_result

ACTIVE = frozenset({"queued", "running"})
WORKER_KINDS = frozenset({"train", "evaluate", "publish"})  # a pipeline is advanced here, never given a worker


@dataclass
class Scheduler:
    jobs: JobStore
    registry: RunRegistry
    runner: Runner = field(default_factory=LocalProcessRunner)
    max_workers: int = 1
    grace: float = 10.0
    running: dict[str, Handle] = field(default_factory=dict)

    def recover(self) -> list[str]:
        """Marks `running` jobs with no live worker `failed`; returns their ids."""
        lost = []
        for job in self.jobs.list(status="running"):
            if job.kind == "pipeline":
                continue  # no worker: its stages are jobs of their own, recovered like any other
            if job.job_id not in self.running and not (job.pid and pid_alive(job.pid)):
                self.jobs.update(job.job_id, status="failed", error="the worker was gone when the scheduler started")
                lost.append(job.job_id)
        return lost

    def tick(self) -> None:
        self.fan_out_experiments()
        self.reap()
        self.advance_pipelines()
        self.claim()

    def run_forever(self, poll_interval: float = 1.0) -> None:
        self.recover()
        try:
            while True:
                self.tick()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            for handle in list(self.running.values()):  # stopping the scheduler stops its workers, cleanly
                self.runner.stop(handle, self.grace)
                self.jobs.update(handle.job_id, status="cancelled", error="the scheduler was stopped")
            raise

    # --- experiments -------------------------------------------------------------------------------------------

    def fan_out_experiments(self) -> list[Job]:
        submitted = []
        for experiment in self.jobs.list_experiments():
            spec = ExperimentSpec.model_validate(experiment.spec)
            completed = self._completed_runs()
            latest: dict[tuple[str, int], Job] = {}
            report: Job | None = None
            for job in self.jobs.list(parent=spec.name):  # oldest first: the newest wins
                if job.kind == "evaluate":
                    report = job
                else:
                    latest[(job.spec.get("tags", {}).get("arm"), job.spec.get("seed"))] = job
            expanded = spec.expand()
            for arm, seed, train in expanded:
                if (spec.name, arm, seed) in completed:
                    continue
                job = latest.get((arm, seed))
                if job is not None and (
                    job.status in ACTIVE
                    or job.status == "succeeded"  # its run is recorded; nothing to redo
                    or (job.finished_at or 0) >= experiment.updated_at  # failed since the last (re)submission
                ):
                    continue
                if train.init_from is not None:
                    parent = (train.init_from.experiment, train.init_from.arm, train.init_from.seed)
                    if parent not in completed:
                        continue  # waits for the run it continues
                submitted.append(self.jobs.submit("train", train.model_dump(mode="json"), parent=spec.name))
            # the report, once every cell is done -- once per (re)submission of the experiment
            done = all((spec.name, arm, seed) in completed for arm, seed, _ in expanded)
            stale = report is None or report.created_at < experiment.updated_at
            if spec.report is not None and done and stale:
                report_spec = spec.report_spec().model_dump(mode="json")
                submitted.append(self.jobs.submit("evaluate", report_spec, parent=spec.name))
        return submitted

    def _completed_runs(self) -> set[tuple[str, str, int]]:
        return {
            (r.config.get("experiment"), r.config.get("arm"), r.config.get("rng_seed"))
            for r in self.registry.list_runs()
            if r.status == "completed" and r.config.get("experiment")
        }

    # --- workers -----------------------------------------------------------------------------------------------

    def reap(self) -> None:
        for job_id, handle in list(self.running.items()):
            job = self.jobs.get(job_id)
            if job.cancel_requested:
                self.runner.stop(handle, self.grace)
                self.jobs.update(job_id, status="cancelled")
                del self.running[job_id]
                continue
            code = self.runner.poll(handle)
            if code is None:
                self.jobs.update(job_id, heartbeat_at=time.time())
                continue
            result = read_result(handle)
            if code == 0 and result is not None:
                self.jobs.update(job_id, status="succeeded", result=result)
            else:
                self.jobs.update(job_id, status="failed", error=f"exit code {code}\n{log_tail(handle.log)}".strip())
            del self.running[job_id]

    def advance_pipelines(self) -> list[Job]:
        """Starts queued pipelines and moves running ones on a stage; returns the stage jobs it submitted."""
        submitted = []
        for pipeline in [
            *self.jobs.list(status="queued", kind="pipeline"),
            *self.jobs.list(status="running", kind="pipeline"),
        ]:
            if pipeline.status == "queued":
                if pipeline.cancel_requested:
                    self.jobs.update(pipeline.job_id, status="cancelled")
                    continue
                pipeline = self.jobs.update(pipeline.job_id, status="running", started_at=time.time())
            spec = PipelineSpec.model_validate(pipeline.spec)
            stages = self.jobs.list(parent=pipeline.job_id)  # oldest first: stage i is the i-th job
            active = [s for s in stages if s.status in ACTIVE]
            if pipeline.cancel_requested:
                for stage in active:
                    self.jobs.request_cancel(stage.job_id)
                if not any(self.jobs.get(s.job_id).status in ACTIVE for s in active):
                    self.jobs.update(pipeline.job_id, status="cancelled")
                continue
            if active:
                continue
            results = {spec.stages[i].name: s.result or {} for i, s in enumerate(stages)}
            if stages and stages[-1].status != "succeeded":
                name = spec.stages[len(stages) - 1].name
                error = f"stage {name} {stages[-1].status}: {stages[-1].error or ''}".strip()
                self.jobs.update(pipeline.job_id, status="failed", error=error, result={"stages": results})
                continue
            if len(stages) == len(spec.stages):
                self.jobs.update(pipeline.job_id, status="succeeded", result={"stages": results})
                continue
            try:
                stage = spec.stage_spec(len(stages), results)
            except ValueError as e:
                self.jobs.update(pipeline.job_id, status="failed", error=str(e), result={"stages": results})
                continue
            submitted.append(self.jobs.submit(stage.kind, stage.model_dump(mode="json"), parent=pipeline.job_id))
        return submitted

    def claim(self) -> None:
        while len(self.running) < self.max_workers:
            job = self.jobs.claim_next(set(WORKER_KINDS))
            if job is None:
                return
            if job.cancel_requested:
                self.jobs.update(job.job_id, status="cancelled")
                continue
            try:
                handle = self.runner.start(job)
            except OSError as e:
                self.jobs.update(job.job_id, status="failed", error=f"couldn't start the worker: {e}")
                continue
            self.jobs.update(job.job_id, pid=handle.pid, log=handle.log)
            self.running[job.job_id] = handle
