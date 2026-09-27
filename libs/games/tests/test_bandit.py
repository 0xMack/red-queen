"""The Rust bandit against its oracle (reference_bandit.py), pull for pull, and the properties the game is built on."""

import copy
import random

import pytest
from reference_bandit import ReferenceBandit

from games.bandit import SCENARIOS, Bandit


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1, 17, 2**31 - 1, 2**63 + 5])
def test_the_rust_game_equals_the_specification_exactly(scenario, seed):
    game, reference = Bandit(scenario, seed), ReferenceBandit(scenario, seed)
    choices = random.Random(seed % 1000 + len(scenario))
    assert (
        (game.arms, game.budget)
        == (reference.k, reference.budget)
        == (SCENARIOS[scenario].arms, SCENARIOS[scenario].budget)
    )
    while not game.done:
        assert game.lamp == reference.lamp
        assert game.means() == reference.means()
        arm = choices.randrange(game.arms)
        assert game.pull(arm) == reference.pull(arm)
    assert (game.total, game.counts, game.best_pulls) == (reference.total, reference.counts, reference.best_pulls)
    assert game.regret == reference.best_expected - reference.expected
    room = reference.best_expected - reference.random_expected
    assert game.skill == ((reference.expected - reference.random_expected) / room if room > 0 else 1.0)


def test_the_nth_pull_of_an_arm_pays_the_same_whatever_else_was_pulled():
    a, b = Bandit("lucky-start", 9), Bandit("lucky-start", 9)
    first = [a.pull(2)[0] for _ in range(20)]
    for arm in (0, 1, 3, 4):
        b.pull(arm)
    assert [b.pull(2)[0] for _ in range(20)] == first


def test_reset_and_copies_replay_the_same_game():
    game = Bandit("two-lamps", 4)
    moves = [game.pull(t % 5) for t in range(30)]
    twin = copy.copy(game)
    game.reset()
    assert [game.pull(t % 5) for t in range(30)] == moves
    assert twin.pulls == 30 and twin.pull(0) == game.pull(0)


def test_regret_and_efficiency():
    best = Bandit("classic", 3)
    arm = best.best_arm
    while not best.done:
        best.pull(arm)
    assert best.regret == 0 and best.efficiency == 1 and best.best_pulls == 100
    worst = Bandit("classic", 3)
    arm = min(range(5), key=lambda a: worst.means()[a])
    while not worst.done:
        worst.pull(arm)
    assert worst.regret > 30 and worst.efficiency < 0.5 and worst.best_pulls == 0


def test_the_reveal_describes_every_arm():
    jackpot = Bandit("jackpot", 1).payouts()
    kinds = sorted(kind for kind, _ in jackpot)
    assert kinds == ["bernoulli", "bernoulli", "bernoulli", "fixed", "jackpot"]
    prize = next(p for kind, p in jackpot if kind == "jackpot")
    assert prize == {"mean": 1.0, "p": 0.02, "prize": 50.0}
    at, after = Bandit("drifting", 1).drift()
    assert 50 <= at <= 70 and len(after) == 5
    assert Bandit("classic", 1).drift() is None


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="scenario"):
        Bandit("roulette", 0)
    game = Bandit("classic", 0)
    with pytest.raises(ValueError, match="no arm"):
        game.pull(5)
    for _ in range(100):
        game.pull(0)
    with pytest.raises(ValueError, match="over"):
        game.pull(0)
