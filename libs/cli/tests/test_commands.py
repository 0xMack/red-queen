"""Every command group against the real backend (its TestClient): what each sends, and what it prints."""

import json

import pytest

from redqueen_cli.client import ApiError

TRAIN = "kind: train\ngame: symbolic-regression\nalgorithm: gp\nbudget: {generations: 2}\n"


def _run(cli, config, status="completed"):
    run_id = cli.registry.create_run(config)
    cli.registry.update_status(run_id, status)
    return run_id


def test_runs_list_show_and_prefixes(cli):
    a = _run(cli, {"game": "checkers", "representation": "td_lambda", "experiment": "e", "arm": "x"})
    _run(cli, {"game": "snake", "representation": "neat"})
    listed = cli("runs", "list", "--game", "checkers")
    assert listed.exit_code == 0, listed.output
    assert (
        a[:8] in listed.output
        and "td_lambda" in listed.output
        and "e/x" in listed.output
        and "neat" not in listed.output
    )

    shown = cli("--json", "runs", "show", a[:6])
    assert json.loads(shown.output)["run_id"] == a
    missing = cli("runs", "show", "zzzz")
    assert isinstance(missing.exception, ApiError) and missing.exception.status == 404


def test_an_ambiguous_prefix_is_refused(cli):
    runs = [_run(cli, {"game": "snake"}) for _ in range(40)]
    common = next(r[:1] for r in runs if sum(o.startswith(r[:1]) for o in runs) > 1)  # 40 ids: a shared first digit
    result = cli("runs", "show", common)
    assert isinstance(result.exception, ApiError) and "matches" in str(result.exception.detail)


def test_jobs_submit_list_show_and_cancel(cli, tmp_path):
    spec = tmp_path / "gp.yaml"
    spec.write_bytes(TRAIN.encode())
    submitted = cli("--json", "jobs", "submit", str(spec), "--set", "budget.generations=5", "--set", "seed=3")
    assert submitted.exit_code == 0, submitted.output
    job = json.loads(submitted.output)
    assert job["status"] == "queued" and job["spec"]["budget"] == {"generations": 5} and job["spec"]["seed"] == 3

    assert job["job_id"][:8] in cli("jobs", "list").output
    assert "gp on symbolic-regression" in cli("jobs", "list").output
    assert json.loads(cli("--json", "jobs", "show", job["job_id"][:6]).output)["job_id"] == job["job_id"]
    cancelled = cli("--json", "jobs", "cancel", job["job_id"][:6])
    assert json.loads(cancelled.output)["status"] == "cancelled"


def test_a_spec_the_api_refuses_comes_back_with_its_reason(cli, tmp_path):
    spec = tmp_path / "bad.yaml"
    spec.write_bytes(TRAIN.replace("gp", "no-such-algorithm").encode())
    result = cli("jobs", "submit", str(spec))
    assert isinstance(result.exception, ApiError) and result.exception.status == 422
    assert "no-such-algorithm" in str(result.exception.detail)


def test_train_builds_a_train_spec(cli):
    parent = _run(cli, {"game": "checkers"})
    result = cli(
        "--json", "train", "td_lambda", "--game", "checkers", "--budget", "iterations=3", "--seed", "7",
        "-p", "agent.hidden=16", "-p", "depth=1", "--init-run", parent[:6], "--held-out-every", "1",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)["spec"]
    assert spec["algorithm"] == "td_lambda" and spec["budget"] == {"iterations": 3} and spec["seed"] == 7
    assert spec["params"] == {"agent": {"hidden": 16}, "depth": 1} and spec["init_from"]["run"] == parent
    assert cli("train", "gp", "--game", "symbolic-regression", "--budget", "lots").exit_code == 2


def test_evaluate_picks_the_games_leaderboard_protocol(cli):
    result = cli("--json", "evaluate", "bandit")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["spec"]["protocol"] == "bandit.skill.v1"
    h2h = cli("--json", "h2h", "material-3", "material-2", "--no-wait")
    assert json.loads(h2h.output)["spec"]["params"]["a"] == "material-3"


def test_leaderboard_show_defaults_to_the_newest_protocol(cli):
    from telemetry import EvaluationRecord, open_stores

    evaluations = open_stores().evaluations
    for protocol, entrant in [("checkers.versus.v9", "baseline:old"), ("checkers.versus.v10", "baseline:new")]:
        evaluations.put(EvaluationRecord(
            game="checkers", protocol=protocol, entrant_id=entrant, entrant_kind="baseline", label=entrant,
            interface="checkers/board32.v1+evaluate1ply.v1", created_at=1.0, metrics={"quality": {"mean": 0.0}},
        ))  # fmt: skip
    shown = cli("--json", "leaderboard", "show", "checkers")
    assert [r["entrant_id"] for r in json.loads(shown.output)] == ["baseline:new"]
    old = cli("--json", "leaderboard", "show", "checkers", "--protocol", "checkers.versus.v9")
    assert [r["entrant_id"] for r in json.loads(old.output)] == ["baseline:old"]


def test_wait_follows_the_log_and_exits_with_the_outcome(cli, monkeypatch, tmp_path):
    """A fake scheduler: the first poll finds the job's worker wrote a log and succeeded."""
    from redqueen_cli import client

    def scheduler_ran_it(_seconds):
        (job,) = cli.jobs.list()
        (tmp_path / "jobs").mkdir(exist_ok=True)
        (tmp_path / "jobs" / "w.log").write_bytes(b"gen 0 best 1.0\n")
        cli.jobs.update(job.job_id, status="succeeded", log="jobs/w.log", result={"run_id": "r1"})

    monkeypatch.setattr(client.time, "sleep", scheduler_ran_it)
    result = cli("evaluate", "bandit", "--wait")
    assert result.exit_code == 0, result.output
    assert "gen 0 best 1.0" in result.output and "succeeded" in result.output and '"run_id": "r1"' in result.output

    def it_failed(_seconds):
        job = next(j for j in cli.jobs.list() if j.status == "queued")
        cli.jobs.update(job.job_id, status="failed", error="exit code 1")

    monkeypatch.setattr(client.time, "sleep", it_failed)
    assert cli("evaluate", "bandit", "--wait").exit_code == 1


EXPERIMENT = """kind: experiment
name: probe
seeds: [0, 1]
defaults: {game: snake, algorithm: neat, budget: {generations: 2}}
arms: {base: {}, big: {train: {params: {population_size: 50}}, baseline: base}}
"""


def test_experiments_submit_status_and_list(cli, tmp_path):
    spec = tmp_path / "probe.yaml"
    spec.write_bytes(EXPERIMENT.encode())
    submitted = cli("experiments", "submit", str(spec))
    assert submitted.exit_code == 0, submitted.output
    assert "0/4 done" in submitted.output
    _run(cli, {"experiment": "probe", "arm": "base", "rng_seed": 0})
    status = cli("experiments", "status", "probe")
    assert "1/4 done" in status.output and "base" in status.output
    assert "probe" in cli("experiments", "list").output
    assert isinstance(cli("experiments", "report", "probe").exception, ApiError)  # no report yet


def test_pipeline_and_schedules(cli, tmp_path):
    stage = "{game: symbolic-regression, algorithm: gp, budget: {generations: 2}}"
    pipeline = tmp_path / "p.yaml"
    pipeline.write_bytes(f"kind: pipeline\nname: p\nstages: [{{name: a, spec: {stage}}}]\n".encode())
    assert json.loads(cli("--json", "pipeline", "submit", str(pipeline)).output)["kind"] == "pipeline"

    schedule = tmp_path / "s.yaml"
    schedule.write_bytes(
        b"kind: schedule\nname: nightly\nevery: 1d\nsubmit: {kind: evaluate, protocol: bandit.skill.v1}\n"
    )
    assert cli("schedules", "submit", str(schedule)).exit_code == 0
    listed = cli("schedules", "list")
    assert "nightly" in listed.output and "bandit.skill.v1" in listed.output
    assert cli("schedules", "delete", "nightly").exit_code == 0
    assert "(none)" in cli("schedules", "list").output


def test_schemas_and_doctor(cli):
    assert "td_lambda" in cli("schemas", "algorithms").output
    doctor = json.loads(cli("--json", "doctor").output)
    assert doctor["api_status"] == "ok" and "rq_alias" in doctor


@pytest.mark.parametrize("bad", ["novalue", "=3"])
def test_set_needs_a_key_and_a_value(cli, tmp_path, bad):
    spec = tmp_path / "gp.yaml"
    spec.write_bytes(TRAIN.encode())
    assert cli("jobs", "submit", str(spec), "--set", bad).exit_code == 2


def test_export_submits_the_frontend_publish_targets(cli):
    run = _run(cli, {"game": "snake", "representation": "q_learning"})
    ballot = json.loads(cli("--json", "export", "ballot", "--no-wait").output)
    assert ballot["kind"] == "publish" and ballot["spec"]["target"] == "frontend.ballot"
    curves = json.loads(cli("--json", "export", "rl-curves", "dqn-snake", "--no-wait").output)
    assert curves["spec"]["params"]["names"] == ["dqn-snake"]
    recording = json.loads(cli("--json", "export", "rl-recording", run[:6], "--label", "x", "--no-wait").output)
    assert recording["spec"]["params"] == {"run": run, "label": "x"}
    pbt = json.loads(cli("--json", "export", "pbt-history", "--seed", "2", "--no-wait").output)
    assert pbt["spec"]["params"]["seed"] == 2 and pbt["spec"]["params"]["arm"] == "g-pbt"


def test_dev_finds_the_repo_root(monkeypatch, tmp_path):
    import typer

    from redqueen_cli.commands.local import repo_root

    root = repo_root()
    assert (root / "libs" / "cli").is_dir()
    monkeypatch.chdir(root / "libs" / "cli" / "src")
    assert repo_root() == root
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.BadParameter):
        repo_root()
