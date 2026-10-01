import checkers_sprt
from games.checkers_openings import ballot
from games.checkers_strategies import STRATEGIES, strategy
from versus_stats import Sprt


def test_the_order_is_the_whole_ballot_shuffled_and_fixed():
    order = checkers_sprt.sprt_order()
    assert sorted(order) == sorted(ballot()) and order != list(ballot())
    assert order == checkers_sprt.sprt_order()


def test_a_much_stronger_player_is_accepted_quickly():
    result = checkers_sprt.sequential_match([(STRATEGIES["material-2"], STRATEGIES["random"])], Sprt(elo1=20))
    assert result["verdict"] == "H1" and result["openings_played"] < 30
    assert result["elo"] > 100 and len(result["per_couple"][0]) == result["openings_played"]


def test_a_player_is_no_stronger_than_itself():
    # A deterministic player against itself: each opening's pair is one win and one loss (or two draws) -- 1 point.
    me = strategy("first-legal")
    result = checkers_sprt.sequential_match([(me, me)], Sprt())
    # decided at the first chance: a zero-variance run of even pairs is as clear as evidence gets
    assert result["verdict"] == "H0" and result["pentanomial"][2] == result["pairs"] == Sprt().min_pairs


def test_couples_take_turns_on_each_opening():
    couples = [(STRATEGIES["material-2"], STRATEGIES["random"]), (STRATEGIES["first-legal"], STRATEGIES["random"])]
    result = checkers_sprt.sequential_match(couples, Sprt(min_pairs=10**6), openings=ballot()[:3])
    assert result["verdict"] == "inconclusive" and result["pairs"] == 6
    assert [len(p) for p in result["per_couple"]] == [3, 3]


def test_across_seeds_is_the_regime_level_view():
    from checkers_selfplay_experiment import across_seeds

    agree = across_seeds({s: {"elo": e} for s, e in enumerate([103.0, 124.0, 127.0, 120.0, 126.0])})
    assert agree["positive"] == 5 and agree["sign_flip_p"] == 0.0625  # the floor with five seeds
    split = across_seeds({s: {"elo": e} for s, e in enumerate([24.0, 21.0, -7.0, 22.0, -33.0])})
    assert split["positive"] == 3 and split["sign_flip_p"] > 0.5
