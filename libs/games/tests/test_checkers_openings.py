import hashlib

import pytest
from evolve import play_match

from games.checkers import Checkers
from games.checkers_openings import all_openings, ballot, level_value


def test_an_opening_is_where_the_game_starts_and_where_reset_returns():
    env = Checkers(opening=(0, 1, 2))
    start = env.reset()
    after = env.board
    assert after != Checkers().board and env.current_player() == 1  # three plies: the second player to move
    env.step(env.legal_moves()[0])
    env.reset()
    assert env.board == after

    # play_match resets its env before the first move: the first position a player sees must be the opening's
    seen = []

    def first_legal(observation, moves):
        seen.append(observation)
        return moves[0]

    play_match(env, {0: first_legal, 1: first_legal})
    assert seen[0] == start


def test_bad_openings_are_rejected():
    with pytest.raises(ValueError):
        Checkers(opening=(99,))


def test_openings_are_distinct_positions():
    openings = all_openings(2)
    positions = {frozenset(Checkers(opening=o).board.items()) for o in openings}
    assert len(positions) == len(openings) == 49


def test_the_ballot_is_level_and_fixed():
    openings = ballot()
    assert len(all_openings(3)) == 216 and len(openings) == 174
    assert level_value(openings[0], 8) == 0.0
    # The ballot is part of the protocol (docs/design/0013): changing it changes every rating. Pin it.
    digest = hashlib.sha256(repr(openings).encode()).hexdigest()[:16]
    assert digest == DIGEST, digest


DIGEST = "ee9a3f188fd685bd"
