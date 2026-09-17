"""Shared rendering utilities for grid-based games.

Deliberately decoupled from evolve.simulation.Environment -- rendering isn't a fitness concern, so
it isn't part of that protocol. A game that wants to be watchable (by a future frontend, a
telemetry trajectory artifact, or this ASCII renderer) implements `render_state()` matching
`Renderable` below; nothing in `evolve` needs to know it exists.
"""

from __future__ import annotations

from typing import Protocol


class Renderable(Protocol):
    def render_state(self) -> dict: ...


def render_grid_ascii(
    width: int,
    height: int,
    cells: dict[tuple[int, int], str],
    symbols: dict[str, str],
    empty: str = ".",
) -> str:
    """Renders a `{(x, y): label}` grid as ASCII text, one character per cell.

    Shared across any grid-based game in this package, not just one -- the whole reason `games` is
    one package instead of one per game (see README.md).
    """
    rows = []
    for y in range(height):
        row = "".join(symbols.get(cells.get((x, y), ""), empty) for x in range(width))
        rows.append(row)
    return "\n".join(rows)
