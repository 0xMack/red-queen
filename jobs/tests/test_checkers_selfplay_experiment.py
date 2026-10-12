"""The self-play experiment's report helpers (jobs/checkers_selfplay_experiment.py)."""


def test_across_seeds_is_the_regime_level_view():
    from checkers_selfplay_experiment import across_seeds

    agree = across_seeds({s: {"elo": e} for s, e in enumerate([103.0, 124.0, 127.0, 120.0, 126.0])})
    assert agree["positive"] == 5 and agree["sign_flip_p"] == 0.0625  # the floor with five seeds
    split = across_seeds({s: {"elo": e} for s, e in enumerate([24.0, 21.0, -7.0, 22.0, -33.0])})
    assert split["positive"] == 3 and split["sign_flip_p"] > 0.5
