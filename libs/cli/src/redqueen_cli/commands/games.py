"""`redqueen leaderboard` and `redqueen models`: what each game's evaluation ranked, and the published model catalog."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from redqueen_cli.output import emit, table
from redqueen_cli.state import state

leaderboard = typer.Typer(help="Leaderboards: every entrant ranked under a protocol.", no_args_is_help=True)
models = typer.Typer(help="Models: published packages, and packaging a run's champion.", no_args_is_help=True)


@leaderboard.command("show")
def show_leaderboard(
    ctx: typer.Context,
    game: str,
    protocol: Annotated[str | None, typer.Option(help="The protocol; the newest one recorded if omitted.")] = None,
) -> None:
    """A game's leaderboard, best first: score (with its 95% interval), entrant kind and interface."""
    st = state(ctx)
    records = st.client.leaderboard(game, protocol)

    def human(rows: list[dict[str, Any]]) -> None:
        ranked = sorted(rows, key=lambda r: r["metrics"].get("quality", {}).get("mean", float("-inf")), reverse=True)
        table(
            [
                [
                    i + 1,
                    r["label"],
                    (q := r["metrics"].get("quality", {})).get("mean"),
                    q.get("ci95"),
                    r["entrant_kind"],
                    r.get("interface") or "",
                    (r.get("run_id") or "")[:8],
                ]
                for i, r in enumerate(ranked)
            ],
            ["#", "ENTRANT", "SCORE", "CI95", "KIND", "INTERFACE", "RUN"],
        )

    emit(records, st.as_json, human)


@leaderboard.command("interfaces")
def interfaces(ctx: typer.Context, game: str) -> None:
    """A game's registered interfaces (observer + action adapter)."""
    st = state(ctx)
    emit(st.client.get(f"/games/{game}/interfaces"), True)


@models.command("catalog")
def catalog(ctx: typer.Context, game: str) -> None:
    """A game's published model packages and their variants."""
    st = state(ctx)
    data = st.client.get(f"/games/{game}/models")

    def human(cat: dict[str, Any]) -> None:
        table(
            [
                [e["label"], ",".join(e["variants"]), (e.get("run_id") or "")[:8], e["package_id"][:12]]
                for e in cat["entries"]
            ],
            ["ENTRANT", "VARIANTS", "RUN", "PACKAGE"],
        )

    emit(data, st.as_json, human)


@models.command("package")
def package(ctx: typer.Context, run: str, ref: str) -> None:
    """Exports a run's champion as a model package on demand (what /watch does for unpublished champions)."""
    st = state(ctx)
    emit(st.client.post(f"/runs/{st.client.resolve_run(run)}/artifacts/{ref}/package"), True)
