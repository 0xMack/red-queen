"""The spec-file groups: `experiments`, `pipeline` and `schedules` -- each a checked-in YAML file submitted to the API,
which the scheduler then runs (docs/design/0018, decision 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from redqueen_cli.output import emit, fields, table, when
from redqueen_cli.state import load_spec, state, submit_job

FileArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="The spec file.")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override a spec field: key.path=value (repeatable).")]

experiments = typer.Typer(help="Experiments: arms x seeds of training runs, then a report.", no_args_is_help=True)
pipeline = typer.Typer(help="Pipelines: stages run one after another.", no_args_is_help=True)
schedules = typer.Typer(help="Schedules: a job submitted every interval or after matching runs.", no_args_is_help=True)


# --- experiments ----------------------------------------------------------------------------------------------------


@experiments.command("submit")
def submit_experiment(ctx: typer.Context, spec: FileArg, set_: SetOpt = None) -> None:
    """Stores (or resubmits: resumes, retrying failed cells) an experiment; the scheduler trains it."""
    st = state(ctx)
    emit(st.client.post("/experiments", json=load_spec(spec, set_)), st.as_json, _matrix)


@experiments.command("list")
def list_experiments(ctx: typer.Context) -> None:
    """Stored experiments."""
    st = state(ctx)
    emit(
        st.client.get("/experiments"),
        st.as_json,
        lambda rows: table([[e["name"], when(e["created_at"])] for e in rows], ["EXPERIMENT", "CREATED"]),
    )


@experiments.command("status")
def experiment_status(ctx: typer.Context, name: str) -> None:
    """The arm x seed matrix: each cell's run (completed), job status, or pending."""
    st = state(ctx)
    emit(st.client.get(f"/experiments/{name}"), st.as_json, _matrix)


_MARK = {"completed": "done", "queued": "queued", "running": "running", "failed": "FAILED", "cancelled": "cancel"}


def _matrix(status: dict[str, Any]) -> None:
    seeds = [str(s) for s in status["seeds"]]
    seeds = sorted({seed for cells in status["arms"].values() for seed in cells} | set(seeds), key=int)
    rows = [
        [
            arm,
            status["baselines"].get(arm, ""),
            *(_MARK.get(cells.get(s, {}).get("status"), "-") if s in cells else "" for s in seeds),
        ]
        for arm, cells in status["arms"].items()
    ]
    table(rows, ["ARM", "BASELINE", *(f"s{s}" for s in seeds)])
    typer.echo(f"{status['done']}/{status['total']} done" + ("; report ready" if status.get("report") else ""))


@experiments.command("report")
def experiment_report(
    ctx: typer.Context,
    name: str,
    h2h: Annotated[bool, typer.Option("--h2h", help="The head-to-head report (experiment.checkers-h2h).")] = False,
    full: Annotated[bool, typer.Option("--full", help="... its full-ballot version.")] = False,
) -> None:
    """The experiment's report, as its report job wrote it (JSON)."""
    st = state(ctx)
    suffix = "-h2h-full" if full else "-h2h" if h2h else ""
    emit(st.client.get(f"/experiments/{name}{suffix}/report"), True)


# --- pipeline -------------------------------------------------------------------------------------------------------


@pipeline.command("submit")
def submit_pipeline(
    ctx: typer.Context,
    spec: FileArg,
    set_: SetOpt = None,
    wait: Annotated[bool, typer.Option("--wait", help="Follow every stage's log until the pipeline ends.")] = False,
) -> None:
    """Submits a pipeline as a job; the scheduler runs its stages in order."""
    submit_job(ctx, load_spec(spec, set_), wait)


# --- schedules ------------------------------------------------------------------------------------------------------


@schedules.command("submit")
def submit_schedule(ctx: typer.Context, spec: FileArg, set_: SetOpt = None) -> None:
    """Stores (or replaces, keeping when it last fired) a schedule."""
    st = state(ctx)
    emit(st.client.post("/schedules", json=load_spec(spec, set_)), st.as_json, fields)


@schedules.command("list")
def list_schedules(ctx: typer.Context) -> None:
    """Stored schedules: their triggers and when each last fired."""
    st = state(ctx)

    def human(rows: list[dict[str, Any]]) -> None:
        table(
            [
                [
                    s["name"],
                    s["spec"].get("every") or "",
                    (s["spec"].get("on_run_completed") or {}).get("game", ""),
                    s["spec"]["submit"].get("protocol") or s["spec"]["submit"].get("kind", "train"),
                    when(s.get("fired_at")),
                ]
                for s in rows
            ],
            ["SCHEDULE", "EVERY", "ON RUN", "SUBMITS", "LAST FIRED"],
        )

    emit(st.client.get("/schedules"), st.as_json, human)


@schedules.command("show")
def show_schedule(ctx: typer.Context, name: str) -> None:
    st = state(ctx)
    emit(st.client.get(f"/schedules/{name}"), st.as_json, fields)


@schedules.command("delete")
def delete_schedule(ctx: typer.Context, name: str) -> None:
    """Stops a schedule firing (the jobs it submitted are kept)."""
    st = state(ctx)
    st.client.delete(f"/schedules/{name}")
    emit({"deleted": name}, st.as_json, lambda d: typer.echo(f"deleted schedule {name}"))


__all__ = ["experiments", "pipeline", "schedules", "submit_job"]
