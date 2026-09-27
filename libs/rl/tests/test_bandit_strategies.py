"""Bandit strategies (docs/design/0011): the Rust update rules against the plain-Python oracle
(reference_bandit_agents.py), pull by pull, and the results each scenario was built to show."""

import math
import statistics

import pytest
from games.bandit import SCENARIOS, Bandit
from reference_bandit_agents import Beliefs

from rl import _native

HINTS = {  # (binary, scale, max payout), as the game hands them to a strategy
    "lucky-start": (False, 3.0, 13.0),
    "jackpot": (False, 7.0, 50.0),
}
PARAMS = {
    "greedy": {},
    "epsilon_greedy": {"epsilon": 0.2, "alpha": 0.3},
    "optimistic": {},
    "ucb1": {"c": 1.2},
    "thompson": {},
    "gradient": {"alpha": 0.25},
    "q_table": {"alpha": 0.4, "epsilon": 0.15},
    "random": {},
}


def _close(a, b):
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("strategy", _native.BANDIT_STRATEGIES)
@pytest.mark.parametrize(
    "scenario,observer",
    [("classic", "none.v1"), ("lucky-start", "none.v1"), ("jackpot", "none.v1"), ("two-lamps", "lamp.v1")],
)
def test_the_rust_beliefs_equal_the_textbook_updates(strategy, scenario, observer):
    params = PARAMS[strategy]
    for seed in (0, 3, 11):
        trace = _native.bandit_trace(strategy, scenario, observer, seed, params)
        game = Bandit(scenario, seed)
        binary, scale, max_payout = HINTS.get(scenario, (True, 1.0, 1.0))
        oracle = Beliefs(strategy, game.arms, 2 if observer == "lamp.v1" else 1, binary, scale, max_payout, params)
        for t, (row, arm, reward, (values, spread, counts, probabilities)) in enumerate(trace):
            # the game the strategy saw is the real game: same lamp, same payout
            assert row == (game.lamp if observer == "lamp.v1" else 0)
            scores = oracle.scores(row)
            if scores is not None:  # a deterministic chooser picked one of its own top-ranked arms
                assert scores[arm] == max(scores), (strategy, t)
            assert game.pull(arm)[0] == reward
            oracle.update(row, arm, reward)
            want_values, want_spread, want_counts, want_probabilities = oracle.expected(row)
            assert list(counts) == want_counts, (strategy, t)
            assert _close(values, want_values), (strategy, t, values, want_values)
            assert _close(spread, want_spread), (strategy, t)
            assert _close(probabilities, want_probabilities), (strategy, t)
        assert game.done and len(trace) == SCENARIOS[scenario].budget


def _mean_regret(strategy: str, scenario: str, observer: str = "none.v1", params=None, games: int = 300) -> float:
    results = _native.bandit_evaluate(strategy, scenario, observer, list(range(games)), params)
    return statistics.fmean(r["regret"] for r in results)


def test_each_scenario_trips_the_strategy_it_was_built_for():
    # Classic: exploring beats committing early.
    assert _mean_regret("thompson", "classic") < 0.7 * _mean_regret("greedy", "classic")
    # Drifting: a constant step size tracks the switch; a sample average stays loyal to the old best arm.
    tracking = {"epsilon": 0.1, "alpha": 0.2}
    assert _mean_regret("epsilon_greedy", "drifting", params=tracking) < _mean_regret("epsilon_greedy", "drifting")
    # Two lamps: a strategy that can't see the lamp can't do better than an average arm.
    assert _mean_regret("thompson", "two-lamps", "lamp.v1") < 0.6 * _mean_regret("thompson", "two-lamps")
    # Drifting: optimism only explores at the start, so it misses the switch that a still-exploring UCB catches (with a
    # bonus sized for 100 pulls -- the textbook c = sqrt(2) explores so much it loses to optimism anyway).
    assert _mean_regret("ucb1", "drifting", params={"c": 0.5}) < _mean_regret("optimistic", "drifting")
    # Too many arms: UCB insists on trying every arm first and spends the budget doing it.
    assert _mean_regret("ucb1", "too-many-arms") > _mean_regret("epsilon_greedy", "too-many-arms")


def test_results_are_reproducible_and_bad_input_is_refused():
    assert _native.bandit_evaluate("ucb1", "jackpot", "none.v1", [4, 5]) == _native.bandit_evaluate(
        "ucb1", "jackpot", "none.v1", [4, 5]
    )
    for bad in [("oracle", "classic", "none.v1"), ("greedy", "roulette", "none.v1"), ("greedy", "classic", "grid.v1")]:
        with pytest.raises(ValueError):
            _native.bandit_evaluate(*bad, [0])
    with pytest.raises(ValueError, match="no parameter"):
        _native.bandit_evaluate("thompson", "classic", "none.v1", [0], {"epsilon": 0.1})
