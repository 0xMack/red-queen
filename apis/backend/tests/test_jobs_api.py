import json

import pytest
from fastapi.testclient import TestClient
from telemetry import SqliteJobStore, SqliteRunRegistry

from backend.dependencies import _job_store, _run_registry
from backend.main import app

TRAIN = {"game": "snake", "algorithm": "neuroevolution", "budget": {"generations": 3}}
EXPERIMENT = {
    "name": "probe-v1",
    "seeds": [0, 1],
    "defaults": {"game": "snake", "algorithm": "neat", "budget": {"generations": 3}},
    "arms": {"base": {}, "big": {"train": {"params": {"population_size": 200}}, "baseline": "base"}},
}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    jobs, registry = SqliteJobStore(tmp_path / "jobs.db"), SqliteRunRegistry(tmp_path / "runs.db")
    app.dependency_overrides[_job_store] = lambda: jobs
    app.dependency_overrides[_run_registry] = lambda: registry
    try:
        yield TestClient(app), jobs, registry, tmp_path
    finally:
        app.dependency_overrides.clear()


def test_a_valid_spec_is_queued_and_an_invalid_one_refused(api):
    client, jobs, _, _ = api
    response = client.post("/jobs", json=TRAIN)
    assert response.status_code == 201
    job = response.json()
    assert job["status"] == "queued" and job["kind"] == "train" and job["spec"]["algorithm"] == "neuroevolution"
    assert jobs.get(job["job_id"]).status == "queued"

    for bad, message in [
        ({**TRAIN, "algorithm": "nope"}, "unknown algorithm"),
        ({**TRAIN, "budget": {"games": 3}}, "budgeted in generations"),
        ({"kind": "experiment", **EXPERIMENT}, "/experiments"),
        ({"kind": "evaluate", "protocol": "nope"}, "unknown protocol"),
    ]:
        refused = client.post("/jobs", json=bad)
        assert refused.status_code == 422 and message in refused.json()["detail"]
    assert len(jobs.list()) == 1  # nothing refused was recorded


def test_jobs_are_listed_filtered_fetched_and_cancelled(api):
    client, jobs, _, _ = api
    train = client.post("/jobs", json=TRAIN).json()
    evaluate = client.post("/jobs", json={"kind": "evaluate", "protocol": "bandit.skill.v1"}).json()
    assert [j["job_id"] for j in client.get("/jobs", params={"kind": "evaluate"}).json()] == [evaluate["job_id"]]
    assert client.get(f"/jobs/{train['job_id']}").json()["kind"] == "train"
    assert client.get("/jobs/nope").status_code == 404

    assert client.post(f"/jobs/{evaluate['job_id']}/cancel").json()["status"] == "cancelled"
    jobs.claim_next()  # the scheduler takes the train job
    marked = client.post(f"/jobs/{train['job_id']}/cancel").json()
    assert marked["status"] == "running" and marked["cancel_requested"]


def test_a_running_jobs_log_is_followed_by_offset(api):
    client, jobs, _, data = api
    job = client.post("/jobs", json=TRAIN).json()
    assert client.get(f"/jobs/{job['job_id']}/logs").json() == {"offset": 0, "text": "", "finished": False}
    (data / "jobs").mkdir()
    log_path = data / "jobs" / f"{job['job_id']}.log"
    log_path.write_bytes(b"gen 0\n")  # bytes: offsets are byte offsets, and text mode writes \r\n on Windows
    jobs.update(job["job_id"], status="running", log=f"jobs/{job['job_id']}.log")
    first = client.get(f"/jobs/{job['job_id']}/logs").json()
    assert first == {"offset": 6, "text": "gen 0\n", "finished": False}
    with open(log_path, "ab") as log:
        log.write(b"gen 1\n")
    jobs.update(job["job_id"], status="succeeded")
    assert client.get(f"/jobs/{job['job_id']}/logs", params={"offset": 6}).json() == {
        "offset": 12,
        "text": "gen 1\n",
        "finished": True,
    }


def test_an_experiment_is_a_matrix_of_runs_jobs_and_pending_cells(api):
    client, jobs, registry, data = api
    created = client.post("/experiments", json=EXPERIMENT)
    assert created.status_code == 201 and created.json()["done"] == 0 and created.json()["total"] == 4

    # base/0 completed as a run (however it was trained); big/0 is a queued job; the rest are pending
    run_id = registry.create_run({"experiment": "probe-v1", "arm": "base", "rng_seed": 0})
    registry.update_status(run_id, "completed")
    jobs.submit("train", {"seed": 0, "tags": {"experiment": "probe-v1", "arm": "big"}}, parent="probe-v1")
    matrix = client.get("/experiments/probe-v1").json()
    assert matrix["arms"]["base"]["0"] == {"status": "completed", "run_id": run_id}
    assert matrix["arms"]["big"]["0"]["status"] == "queued"
    assert matrix["arms"]["base"]["1"] == {"status": "pending"} and matrix["done"] == 1
    assert matrix["baselines"] == {"big": "base"} and matrix["report"] is False

    assert client.get("/experiments/probe-v1/report").status_code == 404
    (data / "experiments").mkdir()
    (data / "experiments" / "probe-v1.json").write_text(json.dumps({"arms": {}}), encoding="utf-8")
    assert client.get("/experiments/probe-v1/report").json() == {"arms": {}}
    assert [e["name"] for e in client.get("/experiments").json()] == ["probe-v1"]
    assert client.get("/experiments/nope").status_code == 404


def test_an_experiment_that_cant_run_is_refused(api):
    client, _, _, _ = api
    bad = {**EXPERIMENT, "arms": {"x": {"train": {"params": {"population_sise": 3}}}}}
    response = client.post("/experiments", json=bad)
    assert response.status_code == 422 and "population_sise" in response.json()["detail"]


def test_schemas_describe_every_kind(api):
    client, _, _, _ = api
    schemas = client.get("/specs/schemas").json()
    assert set(schemas["kinds"]) == {"train", "evaluate", "publish", "experiment", "pipeline"}
    assert "neuroevolution@snake" in schemas["algorithms"] and "checkers.sprt" in schemas["protocols"]
    assert set(schemas["publish_targets"]) == {"snake", "tinylm", "scale-test"}


def test_a_pipeline_is_a_job_and_a_broken_one_is_refused(api):
    client, jobs, _, _ = api
    stage = {"game": "snake", "algorithm": "neat", "budget": {"generations": 2}}
    pipeline = {
        "kind": "pipeline",
        "name": "two",
        "stages": [{"name": "a", "spec": stage}, {"name": "b", "spec": {**stage, "init_from": {"stage": "a"}}}],
    }
    response = client.post("/jobs", json=pipeline)
    assert response.status_code == 201, response.text
    assert response.json()["kind"] == "pipeline" and jobs.get(response.json()["job_id"]).status == "queued"
    broken = {**pipeline, "stages": [{"name": "b", "spec": {**stage, "init_from": {"stage": "a"}}}]}
    assert client.post("/jobs", json=broken).status_code == 422
