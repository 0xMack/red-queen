"""`redqueen jobs`: any job spec submitted, and jobs listed, inspected, followed and cancelled (/jobs)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from redqueen_cli.output import emit, fields, table, when
from redqueen_cli.state import load_spec, state, submit_job

app = typer.Typer(help="Jobs: what the scheduler runs (train, evaluate, publish, pipeline).", no_args_is_help=True)

JobArg = Annotated[str, typer.Argument(help="A job id or a unique prefix of one.")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override a spec field: key.path=value (repeatable).")]
WaitOpt = Annotated[bool, typer.Option("--wait", help="Follow the job's log until it ends; exit 1 unless it succeeds.")]


def resolve_job(ctx: typer.Context, prefix: str) -> str:
    matches = [j["job_id"] for j in state(ctx).client.get("/jobs") if j["job_id"].startswith(prefix)]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"{prefix!r} matches {len(matches)} jobs" + (": give more of the id" if matches else "")
        )
    return matches[0]


@app.command()
def submit(
    ctx: typer.Context,
    spec: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A train/evaluate/publish/pipeline spec.")],
    set_: SetOpt = None,
    wait: WaitOpt = False,
) -> None:
    """Submits a spec file as a job (the API refuses one that can't run, with the reason)."""
    submit_job(ctx, load_spec(spec, set_), wait)


@app.command("list")
def list_jobs(
    ctx: typer.Context,
    status: Annotated[str | None, typer.Option(help="queued, running, succeeded, failed or cancelled.")] = None,
    kind: Annotated[str | None, typer.Option(help="train, evaluate, publish or pipeline.")] = None,
    parent: Annotated[str | None, typer.Option(help="An experiment, pipeline job id, or `schedule:<name>`.")] = None,
    limit: Annotated[int, typer.Option(help="At most this many (newest first); 0 for all.")] = 30,
) -> None:
    """Jobs, newest first."""
    st = state(ctx)
    jobs = list(reversed(st.client.get("/jobs", status=status, kind=kind, parent=parent)))
    jobs = jobs[:limit] if limit else jobs
    emit(jobs, st.as_json, _jobs_table)


def _jobs_table(jobs: list[dict[str, Any]]) -> None:
    table(
        [
            [j["job_id"][:8], j["kind"], j["status"], j.get("parent") or "", _what(j), when(j["created_at"])]
            for j in jobs
        ],
        ["JOB", "KIND", "STATUS", "PARENT", "WHAT", "CREATED"],
    )


def _what(job: dict[str, Any]) -> str:
    spec = job["spec"]
    return {
        "train": lambda: f"{spec.get('algorithm')} on {spec.get('game')}",
        "evaluate": lambda: spec.get("protocol", ""),
        "publish": lambda: spec.get("target", ""),
        "pipeline": lambda: spec.get("name", ""),
    }.get(job["kind"], lambda: "")()


@app.command()
def show(ctx: typer.Context, job: JobArg) -> None:
    """A job: its spec, status, result or error."""
    st = state(ctx)
    emit(st.client.get(f"/jobs/{resolve_job(ctx, job)}"), st.as_json, fields)


@app.command()
def logs(
    ctx: typer.Context,
    job: JobArg,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Keep printing until the job ends.")] = False,
) -> None:
    """A job's worker log (a pipeline's: every stage's)."""
    st = state(ctx)
    job_id = resolve_job(ctx, job)
    if follow:
        finished = st.client.follow_job(job_id, lambda text: typer.echo(text, nl=False))
        typer.echo(f"job {job_id} {finished['status']}", err=True)
        return
    record = st.client.get(f"/jobs/{job_id}")
    ids = [s["job_id"] for s in st.client.get("/jobs", parent=job_id)] if record["kind"] == "pipeline" else [job_id]
    for each in ids:
        typer.echo(st.client.get(f"/jobs/{each}/logs")["text"], nl=False)


@app.command()
def cancel(ctx: typer.Context, job: JobArg) -> None:
    """Cancels a job: a queued one at once, a running one as soon as the scheduler stops its worker."""
    st = state(ctx)
    cancelled = st.client.post(f"/jobs/{resolve_job(ctx, job)}/cancel")
    emit(cancelled, st.as_json, lambda j: typer.echo(f"job {j['job_id']}: {j['status']}"))
