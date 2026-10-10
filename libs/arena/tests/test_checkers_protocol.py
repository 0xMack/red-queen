import itertools

from games.checkers_openings import ballot
from games.checkers_strategies import STRATEGIES

from arena import checkers


def test_a_game_pair_is_one_opening_from_both_seats():
    first, material = STRATEGIES["first-legal"], STRATEGIES["material-2"]
    opening = ballot()[5]
    games = checkers.play_pairing(material, first, [opening], seed_base=0)
    assert games == [
        checkers.play_game(material, first, opening, 0, 0),
        checkers.play_game(material, first, opening, 1, 1),
    ]
    assert checkers.pair_points(games) == [games[0][0] + games[1][0]]


def test_pairings_take_successive_slices_of_the_ballot_wrapping_round_it():
    openings = ballot()
    assert checkers.openings_for(0, 2) == [openings[0], openings[1]]
    assert checkers.openings_for(1, 2) == [openings[2], openings[3]]
    assert checkers.openings_for(len(openings) // 2, 2) == [openings[0], openings[1]]


def test_seed_ranges_are_ten_thousand_apart():
    # Each use of seeded games draws below the next base: a leaderboard run uses 2 * 12 games per pairing, so 10k
    # leaves room for a round robin of ~29 entrants before it would reach the report's range.
    bases = sorted(
        [
            checkers.MONITOR_SEED_BASE,
            checkers.VERSUS_SEED_BASE,
            checkers.REPORT_SEED_BASE,
            checkers.SPRT_SEED_BASE,
            checkers.PBT_SEED_BASE,
        ]
    )
    assert all(b - a == 10_000 for a, b in itertools.pairwise(bases))
