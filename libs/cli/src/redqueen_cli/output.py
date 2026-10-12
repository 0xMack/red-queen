"""How a command prints: `--json` gives the API's JSON unchanged (stable shapes, for scripts and MCP); otherwise a
plain table or key/value block for people."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import typer


def emit(data: Any, as_json: bool, human: Callable[[Any], None] | None = None) -> None:
    if as_json or human is None:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        human(data)


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> None:
    cells = [[_cell(v) for v in row] for row in rows]
    widths = [max([len(h), *(len(r[i]) for r in cells)]) for i, h in enumerate(headers)]
    typer.echo("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip())
    for row in cells:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    if not cells:
        typer.echo("(none)")


def fields(data: dict[str, Any], keys: Sequence[str] | None = None) -> None:
    keys = keys or list(data)
    width = max((len(k) for k in keys), default=0)
    for key in keys:
        value = data.get(key)
        text = json.dumps(value, default=str) if isinstance(value, (dict, list)) else _cell(value)
        typer.echo(f"{key.ljust(width)}  {text}")


def when(timestamp: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp)) if timestamp else ""


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
