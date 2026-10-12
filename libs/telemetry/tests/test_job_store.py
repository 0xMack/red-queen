import threading

import pytest

from telemetry import SqliteJobStore


@pytest.fixture
def store(tmp_path):
    return SqliteJobStore(tmp_path / "jobs.db")


def test_a_job_goes_from_queued_to_running_to_finished(store):
    job = store.submit("train", {"game": "snake"}, parent="exp")
    assert job.status == "queued" and store.get(job.job_id) == job

    claimed = store.claim_next()
    assert claimed.job_id == job.job_id and claimed.status == "running" and claimed.started_at is not None
    assert store.claim_next() is None  # nothing else queued

    done = store.update(job.job_id, status="succeeded", result={"run_id": "abc"})
    assert done.finished_at is not None and done.result == {"run_id": "abc"}
    assert [j.job_id for j in store.list(parent="exp")] == [job.job_id]
    assert store.list(status="queued") == []


def test_claims_are_oldest_first_and_filtered_by_kind(store):
    first = store.submit("evaluate", {})
    second = store.submit("train", {})
    assert store.claim_next({"train"}).job_id == second.job_id
    assert store.claim_next().job_id == first.job_id


def test_concurrent_claimers_never_take_the_same_job(store):
    jobs = {store.submit("train", {"i": i}).job_id for i in range(20)}
    claimed, lock = [], threading.Lock()

    def claimer():
        while (job := store.claim_next()) is not None:
            with lock:
                claimed.append(job.job_id)

    threads = [threading.Thread(target=claimer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(jobs)  # every job once, none twice


def test_cancelling_depends_on_where_the_job_is(store):
    queued = store.submit("train", {})
    assert store.request_cancel(queued.job_id).status == "cancelled"
    running = store.submit("train", {})
    store.claim_next()
    marked = store.request_cancel(running.job_id)
    assert marked.status == "running" and marked.cancel_requested
    store.update(running.job_id, status="cancelled")
    assert store.request_cancel(running.job_id).status == "cancelled"  # finished: left alone


def test_only_known_fields_update_and_unknown_jobs_raise(store):
    job = store.submit("publish", {})
    with pytest.raises(ValueError, match="not updatable"):
        store.update(job.job_id, kind="train")
    with pytest.raises(KeyError):
        store.get("nope")
    with pytest.raises(KeyError):
        store.update("nope", status="failed")


def test_an_experiment_resubmitted_keeps_when_it_began(store):
    first = store.put_experiment("e", {"seeds": [0]})
    again = store.put_experiment("e", {"seeds": [0, 1]})
    assert again.created_at == first.created_at and store.get_experiment("e").spec == {"seeds": [0, 1]}
    assert [e.name for e in store.list_experiments()] == ["e"]
    with pytest.raises(KeyError):
        store.get_experiment("nope")


def test_a_schedule_is_stored_replaced_fired_and_deleted(tmp_path):
    store = SqliteJobStore(tmp_path / "jobs.db")
    first = store.put_schedule("nightly", {"every": "1d"})
    assert first.fired_at is None and store.list_schedules() == [first]
    fired = store.mark_fired("nightly", 123.0)
    assert fired.fired_at == 123.0
    replaced = store.put_schedule("nightly", {"every": "2d"})
    assert replaced.spec == {"every": "2d"} and replaced.fired_at == 123.0 and replaced.created_at == first.created_at
    store.delete_schedule("nightly")
    assert store.list_schedules() == []
    with pytest.raises(KeyError):
        store.get_schedule("nightly")
    with pytest.raises(KeyError):
        store.delete_schedule("nightly")
