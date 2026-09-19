"""Round robin between static Checkers strategies -- the numbers the "Multi-Agent Games" Learn chapter
cites. Every pairing plays GAMES games, alternating seats (so first-move advantage cancels out), each
game seeded (random tie-breaks and random players are reproducible), capped at MAX_PLIES.

Strategies:
- random:       a uniformly random legal move.
- first-legal:  always the first legal move (a deterministic, "dumb but consistent" player).
- material-1:   one-ply lookahead -- the move that leaves the best material balance.
- material-2:   two-ply -- the move whose worst case, after the opponent's material-greedy reply, is
                best (a tiny minimax).

The finding: material-1 is barely better than random, because captures are mandatory -- whenever a
capture exists, *every* legal move is a capture, and otherwise no move changes material, so one ply of
material lookahead has almost nothing to choose between. Looking one ply further (does my move hang a
piece?) wins nearly every game.

Run with: uv run python jobs/checkers_round_robin.py
"""

from __future__ import annotations

import copy
import itertools
import random
from collections.abc import Callable

from evolve import play_match
from games.checkers import Checkers

GAMES = 200
MAX_PLIES = 300

StrategyFactory = Callable[[Checkers, random.Random], Callable]


def random_strategy(env: Checkers, rng: random.Random):
    return lambda observation, moves: rng.choice(moves)


def first_legal(env: Checkers, rng: random.Random):
    return lambda observation, moves: moves[0]


def material_1(env: Checkers, rng: random.Random):
    # simulate() encodes the resulting position from the *opponent's* perspective (it's their move
    # next), so the mover's material is the negated sum.
    return lambda observation, moves: max((-sum(env.simulate(m)), rng.random(), m) for m in moves)[2]


def material_2(env: Checkers, rng: random.Random):
    def pick(observation, moves):
        me = env.current_player()
        best = None
        for move in moves:
            child = copy.deepcopy(env)
            _, _, done = child.step(move)
            if done:
                score = 1000 if child.winner() == me else 0
            else:
                # After the opponent's reply it's my move again, so simulate() is from my perspective;
                # assume they pick the reply that's worst for me.
                score = min(sum(child.simulate(reply)) for reply in child.legal_moves())
            key = (score, rng.random())
            if best is None or key > best[0]:
                best = (key, move)
        return best[1]

    return pick


STRATEGIES: dict[str, StrategyFactory] = {
    "random": random_strategy,
    "first-legal": first_legal,
    "material-1": material_1,
    "material-2": material_2,
}


def play_pairing(a: str, b: str, games: int = GAMES) -> dict[str, int]:
    """Results from `a`'s point of view."""
    tally = {"wins": 0, "draws": 0, "losses": 0}
    for game in range(games):
        rng = random.Random(game)
        env = Checkers()
        seat_a = game % 2
        strategies = {seat_a: STRATEGIES[a](env, rng), 1 - seat_a: STRATEGIES[b](env, rng)}
        result = play_match(env, strategies, max_moves=MAX_PLIES)
        if result.winner is None:
            tally["draws"] += 1
        elif result.winner == seat_a:
            tally["wins"] += 1
        else:
            tally["losses"] += 1
    return tally


def main() -> None:
    for a, b in itertools.combinations(STRATEGIES, 2):
        t = play_pairing(a, b)
        print(f"{a:>12} vs {b:<12} {a}: {t['wins']:3d} W  {t['draws']:3d} D  {t['losses']:3d} L")


if __name__ == "__main__":
    main()
