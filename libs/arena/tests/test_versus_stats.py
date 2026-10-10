import random

import pytest

from arena.versus_stats import (
    Sprt,
    bradley_terry,
    elo_of,
    expected_score,
    pair_elo,
    pentanomial,
    rate,
)


def _pair(rng: random.Random, elo: float, peak_draw_rate: float = 0.4) -> float:
    """One game pair between players `elo` apart, from the first's side: each game a draw (most often between equals),
    else a win with the probability that keeps the expected score at what Elo says."""
    s = expected_score(elo)
    draw_rate = peak_draw_rate * 4 * s * (1 - s)
    win = (s - draw_rate / 2) / (1 - draw_rate)
    points = 0.0
    for _ in range(2):
        if rng.random() < draw_rate:
            points += 0.5
        elif rng.random() < win:
            points += 1.0
    return points


def test_elo_and_expected_score_are_inverse():
    for elo in (-400, -50, 0, 35, 200, 800):
        assert elo_of(expected_score(elo)) == pytest.approx(elo)
    assert expected_score(0) == 0.5 and expected_score(400) == pytest.approx(10 / 11)
    assert elo_of(1.0) < 2000  # a perfect score is capped, not infinite


def test_pentanomial_and_pair_elo():
    pairs = [2.0, 1.5, 1.0, 1.0, 0.5, 2.0]
    assert pentanomial(pairs) == [0, 1, 2, 1, 2]
    estimate = pair_elo(pairs)
    assert estimate["lo"] < estimate["elo"] < estimate["hi"] and estimate["elo"] > 0 and estimate["pairs"] == 6


def test_bradley_terry_matches_expected_to_actual_scores():
    # Without the prior, the maximum-likelihood ratings make every player's expected points equal its actual points.
    results = {("a", "b"): (14.0, 20.0), ("b", "c"): (12.0, 20.0), ("a", "c"): (17.0, 20.0)}
    elo = bradley_terry(results, anchor="c", prior_draws=0.0)
    assert elo["c"] == 0.0 and elo["a"] > elo["b"] > 0
    expected_a = 20 * expected_score(elo["a"] - elo["b"]) + 20 * expected_score(elo["a"] - elo["c"])
    assert expected_a == pytest.approx(14.0 + 17.0, abs=1e-6)


def test_bradley_terry_recovers_true_ratings_and_survives_a_perfect_score():
    rng = random.Random(1)
    true = {"weak": 0.0, "mid": 150.0, "strong": 400.0}
    pairings = {}
    for a, b in [("mid", "weak"), ("strong", "mid"), ("strong", "weak")]:
        pairings[(a, b)] = [_pair(rng, true[a] - true[b]) for _ in range(2000)]
    rated = rate(pairings, anchor="weak", replicates=20)
    for name, elo in true.items():
        assert rated[name]["elo"] == pytest.approx(elo, abs=25), rated
        assert rated[name]["lo"] <= rated[name]["elo"] <= rated[name]["hi"]
    perfect = bradley_terry({("x", "y"): (40.0, 40.0), ("y", "z"): (20.0, 40.0)}, anchor="z")
    assert 300 < perfect["x"] - perfect["y"] < 2000


def test_sprt_accepts_a_real_difference_and_rejects_none():
    rng = random.Random(7)
    stronger = Sprt(elo0=0, elo1=20)
    while stronger.verdict is None and len(stronger.pairs) < 5000:
        stronger.add(_pair(rng, 100.0))
    assert stronger.verdict == "H1" and len(stronger.pairs) < 200

    equal = Sprt(elo0=0, elo1=20)
    while equal.verdict is None and len(equal.pairs) < 20000:
        equal.add(_pair(rng, 0.0))
    assert equal.verdict == "H0"
    summary = equal.summary()
    assert summary["verdict"] == "H0" and summary["bounds"][0] < 0 < summary["bounds"][1]
    assert sum(summary["pentanomial"]) == len(equal.pairs)


def test_sprt_waits_for_its_minimum():
    test = Sprt(min_pairs=10)
    for _ in range(9):
        assert test.add(2.0) is None
    assert test.add(2.0) == "H1"
