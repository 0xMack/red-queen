"""The commands that start one piece of work -- `train`, `evaluate`, `h2h`, `publish` -- each building a spec from
options and submitting it as a job (POST /jobs). A spec file and `redqueen jobs submit` reach every field; these are
the short forms."""

from __future__ import annotations

import re
from typing import Annotated, Any

import typer

from redqueen_cli.client import natural_key
from redqueen_cli.state import apply_sets, state, submit_job

ParamOpt = Annotated[
    list[str] | None, typer.Option("--param", "-p", help="An algorithm/protocol param: key.path=value (repeatable).")
]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Any other spec field: key.path=value (repeatable).")]
WaitOpt = Annotated[bool, typer.Option("--wait", help="Follow the job's log until it ends; exit 1 unless it succeeds.")]


def _with_params(spec: dict[str, Any], params: list[str] | None, sets: list[str] | None) -> dict[str, Any]:
    spec = apply_sets(spec, [f"params.{p}" for p in params or []])
    return apply_sets(spec, sets or [])


def _budget(value: str) -> dict[str, int]:
    unit, sep, count = value.partition("=")
    if not sep or not count.isdigit():
        raise typer.BadParameter(f"--budget takes unit=N (generations=50, iterations=100, ...), not {value!r}")
    return {unit: int(count)}


def train(
    ctx: typer.Context,
    algorithm: Annotated[
        str, typer.Argument(help="A registered algorithm (`redqueen schemas algorithms` lists them).")
    ],
    game: Annotated[str, typer.Option(help="The game (snake, checkers, bandit, reach1d, ...).")],
    budget: Annotated[str, typer.Option(help="unit=N: generations=50, iterations=100, games=200000, steps=3000.")],
    interface: Annotated[str | None, typer.Option(help="A registered interface, if not the game's default.")] = None,
    seed: Annotated[int, typer.Option(help="The run's rng seed.")] = 0,
    held_out_every: Annotated[int | None, typer.Option(help="Log a held-out score every N generations.")] = None,
    init_run: Annotated[str | None, typer.Option(help="Continue this run's final network (an id or prefix).")] = None,
    param: ParamOpt = None,
    set_: SetOpt = None,
    wait: WaitOpt = False,
) -> None:
    """Trains one run: `redqueen train td_lambda --game checkers --budget iterations=200 -p agent.hidden=64`."""
    spec: dict[str, Any] = {"kind": "train", "game": game, "algorithm": algorithm, "budget": _budget(budget)}
    spec["seed"] = seed
    if interface:
        spec["interface"] = interface
    if held_out_every:
        spec["held_out_every"] = held_out_every
    if init_run:
        spec["init_from"] = {"run": state(ctx).client.resolve_run(init_run)}
    submit_job(ctx, _with_params(spec, param, set_), wait)


def evaluate(
    ctx: typer.Context,
    game: Annotated[str, typer.Argument(help="The game whose leaderboard to (re)build.")],
    protocol: Annotated[
        str | None, typer.Option(help="The protocol; the newest version of the game's leaderboard protocol if omitted.")
    ] = None,
    param: ParamOpt = None,
    wait: WaitOpt = False,
) -> None:
    """Re-ranks a game's leaderboard (its leaderboard protocol): `redqueen evaluate checkers --wait`."""
    if protocol is None:
        protocols = state(ctx).client.get("/specs/schemas")["protocols"]
        boards = sorted(n for n, p in protocols.items() if p["game"] == game and p["output"] == "leaderboard")
        # versions of one protocol (checkers.versus.v2, .v3) mean the newest; different protocols need a choice
        if len({re.sub(r"\.v\d+$", "", b) for b in boards}) != 1:
            raise typer.BadParameter(
                f"{game} has {len(boards)} leaderboard protocols ({', '.join(boards) or 'none'}): pick one with --protocol"
            )
        protocol = max(boards, key=natural_key)
    submit_job(ctx, _with_params({"kind": "evaluate", "protocol": protocol}, param, None), wait)


def h2h(
    ctx: typer.Context,
    a: Annotated[str, typer.Argument(help="A fixed strategy (material-6) or a run id prefix, optionally @depth.")],
    b: Annotated[str, typer.Argument(help="The other player, the same way.")],
    elo0: Annotated[float, typer.Option(help="H0: A is no stronger than this.")] = 0.0,
    elo1: Annotated[float, typer.Option(help="H1: A is at least this much stronger.")] = 50.0,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Follow it to the verdict (the default).")] = True,
) -> None:
    """Is Checkers player A stronger than B? Game pairs over the ballot until an SPRT decides (checkers.sprt)."""
    params = {"a": a, "b": b, "elo0": elo0, "elo1": elo1}
    submit_job(ctx, {"kind": "evaluate", "protocol": "checkers.sprt", "params": params}, wait)


def publish(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="snake (its champions), tinylm, or scale-test.")],
    param: ParamOpt = None,
    wait: WaitOpt = False,
) -> None:
    """Packages models into the model store and catalog (ONNX variants, measured parity)."""
    submit_job(ctx, _with_params({"kind": "publish", "target": target}, param, None), wait)
