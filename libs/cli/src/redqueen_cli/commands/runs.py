"""`redqueen runs`: training runs, their curves and champions, and the pause/resume/step control (GET/POST /runs)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from redqueen_cli.output import emit, fields, table, when
from redqueen_cli.state import state

app = typer.Typer(help="Training runs: list, inspect, follow and control them.", no_args_is_help=True)

RunArg = Annotated[str, typer.Argument(help="A run id or a unique prefix of one.")]


def _algorithm(config: dict[str, Any]) -> str:
    return config.get("representation") or config.get("algorithm") or ""


@app.command("list")
def list_runs(
    ctx: typer.Context,
    game: Annotated[str | None, typer.Option(help="Only this game's runs.")] = None,
    status: Annotated[str | None, typer.Option(help="Only runs in this status (running, completed, ...).")] = None,
    experiment: Annotated[str | None, typer.Option(help="Only this experiment's runs.")] = None,
    limit: Annotated[int, typer.Option(help="At most this many (newest first); 0 for all.")] = 30,
) -> None:
    """Runs, newest first: status, game, algorithm, experiment/arm, generations and best fitness."""
    st = state(ctx)
    summaries = {s["run_id"]: s for s in st.client.get("/runs/summaries", trend_points=1)}
    runs = []
    for run in st.client.get("/runs"):
        c = run["config"]
        if (game and c.get("game") != game) or (status and run["status"] != status):
            continue
        if experiment and c.get("experiment") != experiment:
            continue
        s = summaries.get(run["run_id"], {})
        runs.append({**run, "generations": s.get("generations"), "best_fitness": s.get("best_fitness")})
    runs = runs[:limit] if limit else runs

    def human(rows: list[dict[str, Any]]) -> None:
        table(
            [
                [
                    r["run_id"][:8],
                    r["status"],
                    r["config"].get("game", ""),
                    _algorithm(r["config"]),
                    "/".join(str(r["config"][k]) for k in ("experiment", "arm") if r["config"].get(k)),
                    r["generations"],
                    r["best_fitness"],
                    when(r["created_at"]),
                ]
                for r in rows
            ],
            ["RUN", "STATUS", "GAME", "ALGORITHM", "EXPERIMENT", "GENS", "BEST", "CREATED"],
        )

    emit(runs, st.as_json, human)


@app.command()
def show(ctx: typer.Context, run: RunArg) -> None:
    """A run's config, status and summary."""
    st = state(ctx)
    info = st.client.get(f"/runs/{st.client.resolve_run(run)}")
    emit(info, st.as_json, fields)


@app.command()
def history(
    ctx: typer.Context,
    run: RunArg,
    since: Annotated[int, typer.Option(help="From this generation on.")] = 0,
) -> None:
    """Every recorded generation (or RL iteration): best and mean fitness, diversity."""
    st = state(ctx)
    rows = st.client.get(f"/runs/{st.client.resolve_run(run)}/metrics/history", since_generation=since or None)
    emit(rows, st.as_json, _history_table)


def _history_table(rows: list[dict[str, Any]]) -> None:
    table(
        [[r["generation"], r["best_fitness"], r["mean_fitness"], r.get("diversity")] for r in rows],
        ["GEN", "BEST", "MEAN", "DIVERSITY"],
    )


@app.command()
def tail(
    ctx: typer.Context,
    run: RunArg,
    interval: Annotated[float, typer.Option(help="Seconds between polls.")] = 2.0,
) -> None:
    """Prints new generations as they're recorded, until the run stops running."""
    st = state(ctx)
    run_id, since = st.client.resolve_run(run), 0
    while True:
        rows = st.client.get(f"/runs/{run_id}/metrics/history", since_generation=since or None)
        for r in rows:
            if st.as_json:
                typer.echo(json.dumps(r))
            else:
                typer.echo(f"gen {r['generation']:>5}  best {r['best_fitness']:.4g}  mean {r['mean_fitness']:.4g}")
        if rows:
            since = rows[-1]["generation"] + 1
        if st.client.get(f"/runs/{run_id}")["status"] not in {"running", "paused"}:
            return
        time.sleep(interval)


def _control(ctx: typer.Context, run: str, action: str) -> None:
    st = state(ctx)
    run_id = st.client.resolve_run(run)
    st.client.post(f"/runs/{run_id}/control", json={"action": action})
    emit({"run_id": run_id, "action": action}, st.as_json, lambda d: typer.echo(f"{action}: {run_id}"))


@app.command()
def pause(ctx: typer.Context, run: RunArg) -> None:
    """Pauses a running job's training loop between generations."""
    _control(ctx, run, "pause")


@app.command()
def resume(ctx: typer.Context, run: RunArg) -> None:
    """Resumes a paused run."""
    _control(ctx, run, "resume")


@app.command()
def step(ctx: typer.Context, run: RunArg) -> None:
    """Lets a paused run train exactly one more generation."""
    _control(ctx, run, "step")


@app.command()
def artifact(
    ctx: typer.Context,
    run: RunArg,
    ref: Annotated[str, typer.Argument(help="An artifact ref (a champion ref, `<run>-gen<N>`).")],
    kind: Annotated[str | None, typer.Option(help="The artifact kind, if not the default.")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write it here instead of printing it.")] = None,
) -> None:
    """A stored artifact (a champion's genome or policy) as JSON."""
    st = state(ctx)
    data = st.client.get(f"/runs/{st.client.resolve_run(run)}/artifacts/{ref}", kind=kind)
    _dump(data, out)


@app.command()
def brain(
    ctx: typer.Context,
    run: RunArg,
    ref: Annotated[str, typer.Argument(help="A champion ref.")],
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """A champion compiled to plain numbers -- what the browser plays (a Checkers evaluator, a network's layers)."""
    st = state(ctx)
    _dump(st.client.get(f"/runs/{st.client.resolve_run(run)}/artifacts/{ref}/brain"), out)


def _dump(data: Any, out: Path | None) -> None:
    text = json.dumps(data, indent=2)
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)
