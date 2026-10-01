"""Checkers openings for measuring strength (docs/design/0013): the ballot.

Two deterministic players replay the same game from the standard start, so a comparison between them needs varied
starting positions. Tournament checkers faces the same problem between strong humans and solves it with a *three-move
ballot*: the first three moves are drawn from a list of openings known to be playable for both sides. This is the same
idea derived from this project's rules, not the official list: every position three plies from the start
(transpositions counted once), kept only if material search `BALLOT_DEPTH` plies deep scores it as level for the side
to move. A player that loses from a ballot opening lost it over the board.

An opening is a tuple of move indices into `Checkers.legal_moves()`, played from the start -- move order is part of
the specified game (docs/design/0009), so an index sequence is the same opening everywhere. `Checkers(opening=...)`
starts a game there.
"""

from __future__ import annotations

from functools import cache

from games import _native
from games.checkers import Checkers

Opening = tuple[int, ...]

BALLOT_PLIES = 3
BALLOT_DEPTH = 8


def all_openings(plies: int) -> list[Opening]:
    """Every position `plies` moves from the start, as the lexicographically first move sequence reaching it (two
    sequences reaching the same position -- the same pieces on the same squares, the same side to move -- are one
    opening)."""
    found: dict[tuple, Opening] = {}

    def visit(sequence: Opening) -> None:
        env = Checkers(opening=sequence)
        if len(sequence) == plies:
            found.setdefault((frozenset(env.board.items()), env.current_player()), sequence)
            return
        for index in range(len(env.legal_moves())):
            visit((*sequence, index))

    visit(())
    return sorted(found.values())


def level_value(opening: Opening, depth: int) -> float:
    """Material search's value of the opening's position for the side to move, `depth` plies deep (men 1, kings 2;
    0 is level)."""
    env = Checkers(opening=opening)
    search = _native.CheckersStrategy(f"material-{depth}", 0, None, None, 1, None)
    return max(search.scores(env._core))


@cache
def ballot(plies: int = BALLOT_PLIES, depth: int = BALLOT_DEPTH) -> tuple[Opening, ...]:
    """The openings every strength measurement plays (174 with the defaults), in a fixed order."""
    return tuple(opening for opening in all_openings(plies) if level_value(opening, depth) == 0.0)
