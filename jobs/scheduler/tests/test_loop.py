"""The scheduler's decisions, with a runner that only pretends: what gets claimed, submitted, reaped and recovered."""

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from telemetry import SqliteJobStore, SqliteRunRegistry

from scheduler import Scheduler
from scheduler.runner import Handle

EXPERIMENT = {
    "kind": "experiment",
    "name": "probe",
    "seeds": [0, 1],
    "defaults": {"game": "snake", "algorithm": "neat", "budget": {"generations": 2}},
    "arms": {"base": {}, "ft": {"baseline": "base", "init_from": "probe/base"}},
}


@dataclass
class FakeRunner:
    """Workers that exit when told to: `finish(job_id, code, result)`."""

    exits: dict[str, tuple[int, dict | None]] = field(default_factory=dict)
    started: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    pids: itertools.count = field(default_factory=lambda: itertools.count(1000))

    def start(self, job):
        self.started.append(job.job_id)
        return Handle(job.job_id, next(self.pids), f"jobs/{job.job_id}.log", Path("nowhere"))

    def poll(self, handle):
        return self.exits.get(handle.job_id, (None, None))[0]

    def stop(self, handle, grace):
        self.stopped.append(handle.job_id)

    def finish(self, job_id, code=0, result=None):
        self.exits[job_id] = (code, result)


@pytest.fixture
def world(tmp_path, monkeypatch):
    jobs, registry, runner = SqliteJobStore(tmp_path / "jobs.db"), SqliteRunRegistry(tmp_path / "runs.db"), FakeRunner()
    monkeypatch.setattr("scheduler.loop.read_result", lambda handle: runner.exits.get(handle.job_id, (0, None))[1])
    return Scheduler(jobs, registry, runner, max_workers=2), jobs, registry, runner


def _complete(registry, arm, seed):
    run_id = registry.create_run({"experiment": "probe", "arm": arm, "rng_seed": seed})
    registry.update_status(run_id, "completed")
    return run_id


def test_queued_jobs_are_claimed_up_to_the_worker_limit_and_reaped(world):
    scheduler, jobs, _, runner = world
    a, b, c = (jobs.submit("train", {"n": i}) for i in range(3))
    scheduler.tick()
    assert runner.started == [a.job_id, b.job_id] and jobs.get(c.job_id).status == "queued"
    assert jobs.get(a.job_id).pid is not None and jobs.get(a.job_id).log == f"jobs/{a.job_id}.log"

    runner.finish(a.job_id, 0, {"run_id": "r1"})
    runner.finish(b.job_id, 3)
    scheduler.tick()
    assert jobs.get(a.job_id).status == "succeeded" and jobs.get(a.job_id).result == {"run_id": "r1"}
    assert jobs.get(b.job_id).status == "failed" and jobs.get(b.job_id).error.startswith("exit code 3")
    assert runner.started[-1] == c.job_id  # a freed slot takes the next queued job


def test_a_cancel_stops_the_worker(world):
    scheduler, jobs, _, runner = world
    job = jobs.submit("train", {})
    scheduler.tick()
    jobs.request_cancel(job.job_id)
    scheduler.tick()
    assert runner.stopped == [job.job_id] and jobs.get(job.job_id).status == "cancelled"


def test_an_experiment_fans_out_and_a_continuing_arm_waits_for_its_parent(world):
    scheduler, jobs, registry, _ = world
    jobs.put_experiment("probe", EXPERIMENT)
    submitted = scheduler.fan_out_experiments()
    # base for both seeds; ft waits until base's run of the same seed exists
    assert [(j.spec["tags"]["arm"], j.spec["seed"]) for j in submitted] == [("base", 0), ("base", 1)]
    assert all(j.parent == "probe" for j in submitted)
    assert scheduler.fan_out_experiments() == []  # in flight: nothing new

    _complete(registry, "base", 0)
    ft = scheduler.fan_out_experiments()
    assert [(j.spec["tags"]["arm"], j.spec["seed"]) for j in ft] == [("ft", 0)]
    assert ft[0].spec["init_from"] == {"run": None, "experiment": "probe", "arm": "base", "seed": 0}


def test_a_completed_cell_is_never_retrained_and_a_failed_one_waits_for_a_resubmission(world):
    scheduler, jobs, registry, runner = world
    _complete(registry, "base", 0)  # trained before the scheduler existed: counts
    jobs.put_experiment("probe", {**EXPERIMENT, "seeds": [0]})
    (ft,) = scheduler.fan_out_experiments()
    assert ft.spec["tags"]["arm"] == "ft"

    scheduler.claim()
    runner.finish(ft.job_id, 1)
    scheduler.reap()
    assert scheduler.fan_out_experiments() == []  # failed since the last submission: not retried by itself
    jobs.put_experiment("probe", {**EXPERIMENT, "seeds": [0]})  # resubmitted: retried
    (retry,) = scheduler.fan_out_experiments()
    assert retry.spec["tags"]["arm"] == "ft" and retry.job_id != ft.job_id


def test_recover_fails_running_jobs_whose_worker_is_gone(world):
    scheduler, jobs, _, _ = world
    job = jobs.submit("train", {})
    jobs.claim_next()
    jobs.update(job.job_id, pid=2**22 + 12345)  # no such process
    assert scheduler.recover() == [job.job_id]
    assert jobs.get(job.job_id).status == "failed" and "gone" in jobs.get(job.job_id).error


def test_an_arm_can_run_its_own_seeds(world):
    scheduler, jobs, _, _ = world
    jobs.put_experiment("probe", {**EXPERIMENT, "arms": {"base": {}, "more": {"seeds": [0, 1, 2]}}})
    submitted = scheduler.fan_out_experiments()
    assert sorted((j.spec["tags"]["arm"], j.spec["seed"]) for j in submitted) == [
        ("base", 0),
        ("base", 1),
        ("more", 0),
        ("more", 1),
        ("more", 2),
    ]


def test_the_report_runs_once_every_cell_is_done_and_again_after_a_resubmission(world):
    scheduler, jobs, registry, _ = world
    spec = {**EXPERIMENT, "seeds": [0], "arms": {"base": {}}, "report": {"protocol": "experiment.snake-report"}}
    jobs.put_experiment("probe", spec)
    scheduler.fan_out_experiments()
    _complete(registry, "base", 0)
    (report,) = scheduler.fan_out_experiments()
    assert report.kind == "evaluate" and report.parent == "probe"
    assert report.spec["protocol"] == "experiment.snake-report" and report.spec["params"] == {"experiment": "probe"}
    assert scheduler.fan_out_experiments() == []  # once
    jobs.put_experiment("probe", spec)
    (again,) = scheduler.fan_out_experiments()
    assert again.kind == "evaluate" and again.job_id != report.job_id


def test_submit_stores_an_experiment_file(tmp_path, monkeypatch):
    from telemetry import open_job_store

    from scheduler.__main__ import main

    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    path = tmp_path / "probe.yaml"
    path.write_bytes(
        b"kind: experiment\nname: probe\nseeds: [0]\ndefaults: {game: snake, algorithm: neat, budget: {generations: 1}}\narms: {a: {}}\n"
    )
    main(["--submit", str(path), "--once"])
    (experiment,) = open_job_store().list_experiments()
    assert experiment.name == "probe"
    assert [j.spec["tags"]["arm"] for j in open_job_store().list(parent="probe")] == ["a"]


PIPELINE = {
    "kind": "pipeline",
    "name": "recipe",
    "stages": [
        {"name": "td", "spec": {"game": "checkers", "algorithm": "td_lambda", "budget": {"iterations": 2}}},
        {
            "name": "leaf",
            "spec": {
                "game": "checkers",
                "algorithm": "td_lambda",
                "budget": {"iterations": 1},
                "init_from": {"stage": "td"},
            },
        },
    ],
}


def test_a_pipeline_runs_its_stages_in_order_each_continuing_the_last(world):
    scheduler, jobs, _, runner = world
    pipeline = jobs.submit("pipeline", PIPELINE)
    scheduler.tick()
    (td,) = jobs.list(parent=pipeline.job_id)
    assert jobs.get(pipeline.job_id).status == "running" and runner.started == [td.job_id]  # the pipeline: no worker
    scheduler.tick()
    assert len(jobs.list(parent=pipeline.job_id)) == 1  # the next stage waits

    runner.finish(td.job_id, 0, {"run_id": "td-run"})
    scheduler.tick()
    _, leaf = jobs.list(parent=pipeline.job_id)
    assert leaf.spec["init_from"] == {"run": "td-run", "experiment": None, "arm": None, "seed": None}
    runner.finish(leaf.job_id, 0, {"run_id": "leaf-run"})
    scheduler.tick()
    scheduler.tick()
    done = jobs.get(pipeline.job_id)
    assert done.status == "succeeded"
    assert done.result == {"stages": {"td": {"run_id": "td-run"}, "leaf": {"run_id": "leaf-run"}}}


def test_a_pipeline_fails_with_its_first_failing_stage_and_a_cancel_stops_the_running_one(world):
    scheduler, jobs, _, runner = world
    failing = jobs.submit("pipeline", PIPELINE)
    scheduler.tick()
    (td,) = jobs.list(parent=failing.job_id)
    runner.finish(td.job_id, 2)
    scheduler.tick()
    scheduler.tick()
    assert jobs.get(failing.job_id).status == "failed" and jobs.get(failing.job_id).error.startswith("stage td failed")
    assert len(jobs.list(parent=failing.job_id)) == 1

    cancelled = jobs.submit("pipeline", PIPELINE)
    scheduler.tick()
    (td,) = jobs.list(parent=cancelled.job_id)
    jobs.request_cancel(cancelled.job_id)
    scheduler.tick()
    scheduler.tick()
    assert td.job_id in runner.stopped and jobs.get(td.job_id).status == "cancelled"
    assert jobs.get(cancelled.job_id).status == "cancelled"


def test_recover_leaves_a_running_pipeline_alone(world):
    scheduler, jobs, _, _ = world
    pipeline = jobs.submit("pipeline", PIPELINE)
    scheduler.advance_pipelines()
    assert scheduler.recover() == [] and jobs.get(pipeline.job_id).status == "running"


RERANK = {"kind": "schedule", "name": "rerank", "submit": {"kind": "evaluate", "protocol": "checkers.versus.v2"}}


def test_an_interval_schedule_fires_when_due_and_never_piles_up(world):
    scheduler, jobs, _, _ = world
    stored = jobs.put_schedule("rerank", {**RERANK, "every": "1h"})
    assert scheduler.fire_schedules(now=stored.updated_at + 60) == []  # not yet an hour since it was submitted
    (job,) = scheduler.fire_schedules(now=stored.updated_at + 3600)
    assert job.kind == "evaluate" and job.parent == "schedule:rerank"
    assert scheduler.fire_schedules(now=stored.updated_at + 9 * 3600) == []  # its last job is still queued
    jobs.update(job.job_id, status="succeeded")
    assert scheduler.fire_schedules(now=stored.updated_at + 9 * 3600) != []


def test_a_run_triggered_schedule_fires_once_per_new_matching_run(world):
    scheduler, jobs, registry, _ = world
    before = registry.create_run({"game": "checkers"})
    registry.update_status(before, "completed")
    jobs.put_schedule("rerank", {**RERANK, "on_run_completed": {"game": "checkers"}})
    assert scheduler.fire_schedules() == []  # completed before the schedule existed

    for config in ({"game": "snake"}, {"game": "checkers", "experiment": "e", "arm": "a"}):  # not entrants of its game
        registry.update_status(registry.create_run(config), "completed")
    assert scheduler.fire_schedules() == []
    registry.create_run({"game": "checkers"})  # still running
    assert scheduler.fire_schedules() == []

    time.sleep(0.01)
    registry.update_status(registry.create_run({"game": "checkers"}), "completed")
    (job,) = scheduler.fire_schedules()
    jobs.update(job.job_id, status="succeeded")
    assert scheduler.fire_schedules() == []  # that run has been counted
