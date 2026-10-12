"""`redqueen` (alias `rq`): the backend's API from the command line (docs/design/0018, decisions 1 and 5).

Command groups follow the API's schema slices (runs, jobs, experiments, ...), and verbs wrap work that takes several
calls (submit a job, then follow its log to the end). Every command takes `--json` for stable machine output; every
command that starts work takes `--wait`. The API is `--api` / REDQUEEN_API_URL (default http://localhost:8000).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from redqueen_cli.client import ApiError
from redqueen_cli.commands import games, jobs, runs, specs, work
from redqueen_cli.output import emit, table
from redqueen_cli.state import State, state

app = typer.Typer(
    name="redqueen",
    help=__doc__.split("\n\n")[1].replace("\n", " "),
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main(
    ctx: typer.Context,
    api: Annotated[str | None, typer.Option(envvar="REDQUEEN_API_URL", help="The API's base URL.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine output: the API's JSON, unchanged.")] = False,
) -> None:
    st = ctx.ensure_object(State)
    st.api_url = api or st.api_url
    st.as_json = as_json or st.as_json


app.add_typer(runs.app, name="runs")
app.add_typer(jobs.app, name="jobs")
app.add_typer(specs.experiments, name="experiments")
app.add_typer(specs.pipeline, name="pipeline")
app.add_typer(specs.schedules, name="schedules")
app.add_typer(games.leaderboard, name="leaderboard")
app.add_typer(games.models, name="models")
app.command()(work.train)
app.command()(work.evaluate)
app.command()(work.h2h)
app.command()(work.publish)


@app.command()
def schemas(
    ctx: typer.Context,
    section: Annotated[
        str | None, typer.Argument(help="kinds, algorithms, protocols or publish_targets (default: their names).")
    ] = None,
) -> None:
    """What the API accepts: every spec kind, algorithm, protocol and publish target, with its params' JSON Schema."""
    st = state(ctx)
    data = st.client.get("/specs/schemas")
    if section is not None:
        if section not in data:
            raise typer.BadParameter(f"one of {', '.join(data)}")
        emit(data[section], st.as_json, lambda d: table([[k, _summary(v)] for k, v in d.items()], ["NAME", "SUMMARY"]))
        return
    emit(data, st.as_json, lambda d: [typer.echo(f"{k}: {', '.join(v)}") for k, v in d.items()])


def _summary(entry: Any) -> str:
    if isinstance(entry, dict):
        games_ = entry.get("games") or ([entry["game"]] if entry.get("game") else [])
        where = f" [{', '.join(games_)}]" if games_ else ""
        return (entry.get("summary") or entry.get("title") or entry.get("description") or "") + where
    return ""


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Checks the setup: the API is reachable, and whether `rq` on PATH is this CLI (it's a common name)."""
    st = state(ctx)
    checks: dict[str, Any] = {"api": st.client.base_url}
    try:
        checks["api_status"] = st.client.get("/health").get("status")
    except ApiError as e:
        checks["api_status"] = f"unreachable: {e.detail}"
    checks.update(rq_check())
    emit(checks, st.as_json, lambda c: [typer.echo(f"{k:<12} {v}") for k, v in c.items()])


def rq_check() -> dict[str, str]:
    """Where `redqueen` and `rq` resolve on PATH, and whether `rq` is ours -- outside the venv, PATH order decides
    silently, and RQ (Redis Queue) also installs an `rq` (docs/design/0018, decision 8)."""
    redqueen, rq = shutil.which("redqueen"), shutil.which("rq")
    # our scripts' directory: where `redqueen` resolves, or where this process was started from
    ours = {Path(p).resolve().parent for p in (redqueen, sys.argv[0] if sys.argv else None) if p}
    out = {"redqueen": redqueen or "not on PATH (use `uv run redqueen`)", "rq": rq or "not on PATH"}
    if rq is None:
        out["rq_alias"] = "unavailable: call `redqueen`"
    elif Path(rq).resolve().parent in ours:
        out["rq_alias"] = "ok: `rq` is this CLI"
    else:
        out["rq_alias"] = f"SHADOWED: `rq` is {rq}, another tool -- call `redqueen` instead"
    return out


def run() -> None:
    """The console-script entry point: API errors become one line on stderr and exit code 1, not a traceback."""
    for stream in (sys.stdout, sys.stderr):
        # entrant labels have arrows and Greek letters: piped output is UTF-8, and a console that can't show a
        # character (cp1252) gets a `?` rather than a UnicodeEncodeError
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(**({"errors": "replace"} if stream.isatty() else {"encoding": "utf-8"}))
    try:
        app()
    except ApiError as e:
        typer.echo(f"error: {e.detail}" if e.status in (0, 422) else f"error {e.status}: {e.detail}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    run()
