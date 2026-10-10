"""Checkers' evaluation protocols (docs/design/0007 "Versus", 0013) and playing two strategies under them.

Protocol `checkers.versus.v2`: a round robin over the **ballot** (games.checkers_openings: the level three-ply
openings). Every pair of entrants plays OPENINGS_PER_PAIR openings, each as a **game pair** -- once from each seat --
so opening luck cancels; each pairing takes the next openings in the ballot, so a round robin covers all of it. Games
are seeded (random tie-breaks reproduce) and capped at MAX_PLIES. An entrant's score is a Bradley-Terry Elo rating
(`arena.versus_stats`), anchored so ANCHOR is 0.

Every use of seeded Checkers games owns a seed range, listed here so they stay disjoint: training opponents draw by
opponent index, monitoring from MONITOR_SEED_BASE, the leaderboard from VERSUS_SEED_BASE, the self-play experiment
report from REPORT_SEED_BASE, head-to-head SPRTs from SPRT_SEED_BASE. Changing any protocol constant means a new
protocol version, not an edit.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from evolve import play_match
from games.checkers import Checkers
from games.checkers_openings import Opening, ballot

GAME = "checkers"
PROTOCOL = "checkers.versus.v2"
INTERFACE = "checkers/board32.v1+evaluate1ply.v1"
OPENINGS_PER_PAIR = 12  # 24 games per pairing, one game pair per opening
MAX_PLIES = 300
ANCHOR = "baseline:random"  # rated 0

MONITOR_SEED_BASE = 20_000
VERSUS_SEED_BASE = 30_000
REPORT_SEED_BASE = 40_000
SPRT_SEED_BASE = 50_000
SPRT_ORDER_SEED = 0  # the fixed shuffle of the ballot a sequential test plays through

# (env, rng) -> Strategy
Factory = Callable[[Checkers, random.Random], Callable]


def play_game(a: Factory, b: Factory, opening: Opening, seat_a: int, seed: int) -> tuple[float, int]:
    """One game from `opening`, `a` in `seat_a` (0 moves first from the standard start): (a's points, plies)."""
    env = Checkers(opening=opening)
    strategies = {
        seat_a: a(env, random.Random(seed)),
        1 - seat_a: b(env, random.Random(seed + 7919)),
    }
    match = play_match(env, strategies, max_moves=MAX_PLIES)
    points = 0.5 if match.winner is None else 1.0 if match.winner == seat_a else 0.0
    return points, match.moves_played


def play_pairing(a: Factory, b: Factory, openings: list[Opening], seed_base: int) -> list[tuple[float, int]]:
    """A game pair per opening -- `a` from seat 0, then from seat 1 -- as a flat list of (a's points, plies): games
    2k and 2k + 1 are opening k's pair."""
    results = []
    for k, opening in enumerate(openings):
        for seat_a in (0, 1):
            results.append(play_game(a, b, opening, seat_a, seed_base + 2 * k + seat_a))
    return results


def pair_points(games: list[tuple[float, int]]) -> list[float]:
    """The points of each game pair in `play_pairing`'s output (0 ... 2)."""
    return [games[i][0] + games[i + 1][0] for i in range(0, len(games) - 1, 2)]


def openings_for(index: int, per_pair: int) -> list[Opening]:
    """The openings pairing `index` plays: the next `per_pair` of the ballot, wrapping, so a round robin covers it."""
    openings = ballot()
    return [openings[(index * per_pair + k) % len(openings)] for k in range(per_pair)]
