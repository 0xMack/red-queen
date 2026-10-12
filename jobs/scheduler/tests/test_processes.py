"""Real workers: a job run end to end in a child process, and one cancelled mid-run -- stopped cleanly, so its run is
marked `failed` rather than left `running` forever (CTRL_BREAK_EVENT on Windows, SIGTERM elsewhere; jobcore.cli)."""

import time

from telemetry import data_dir, open_job_store, open_stores

from scheduler import Scheduler

GP = {"kind": "train", "game": "symbolic-regression", "algorithm": "gp", "budget": {"generations": 3}}
LONG = {
    "kind": "train",
    "game": "checkers",
    "algorithm": "td_lambda",
    "budget": {"iterations": 10_000},
    "params": {"games_per_iteration": 20, "depth": 1, "monitor_games": 2, "agent": {"hidden": 8}},
}


def _until(predicate, scheduler, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        scheduler.tick()
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_a_job_runs_in_a_worker_process_and_its_run_is_recorded():
    jobs, registry = open_job_store(), open_stores().registry
    scheduler = Scheduler(jobs, registry)
    job = jobs.submit("train", GP)
    _until(lambda: jobs.get(job.job_id).status not in {"queued", "running"}, scheduler)

    done = jobs.get(job.job_id)
    assert done.status == "succeeded", done.error
    run = registry.get_run(done.result["run_id"])
    assert run.status == "completed" and run.config["representation"] == "linear_gp"
    assert done.result["run_id"] in (data_dir() / done.log).read_text(encoding="utf-8")


def test_a_cancelled_worker_stops_and_its_run_is_marked_failed():
    jobs, registry = open_job_store(), open_stores().registry
    scheduler = Scheduler(jobs, registry, grace=30.0)
    job = jobs.submit("train", LONG)
    # wait until the worker has created its run (it is training), then cancel
    _until(lambda: any(r.status == "running" for r in registry.list_runs()), scheduler)
    jobs.request_cancel(job.job_id)
    _until(lambda: jobs.get(job.job_id).status == "cancelled", scheduler)

    (run,) = registry.list_runs()
    deadline = time.monotonic() + 30
    while run.status == "running" and time.monotonic() < deadline:  # the worker's last act is marking it
        time.sleep(0.2)
        run = registry.get_run(run.run_id)
    assert run.status == "failed"
    # stopped by the signal (raised as KeyboardInterrupt, re-raised by recorded_run), not by some crash
    assert "KeyboardInterrupt" in (data_dir() / jobs.get(job.job_id).log).read_text(encoding="utf-8")
