"""Tabular Q-learning / SARSA: the Rust update rule against the plain-Python oracle (reference_tabular.py), exactly;
and the agents learning through the real Snake and Reach1D adapters."""

import json
import random
import statistics

import pytest
from reference_tabular import replay

import rl
from rl import _native

FEATURES = "snake/features.v1+relative3.v1"


def _episodes(rng: random.Random, states: int, actions: int, count: int, sarsa: bool) -> list[tuple]:
    """Random episodes: chained states, rewards in [-1, 1], ending in a game over or a truncation."""
    out = []
    for _ in range(count):
        state = rng.randrange(states)
        action = rng.randrange(actions)
        for t in range(rng.randint(1, 12)):
            last = t == 11 or rng.random() < 0.15
            next_state = rng.randrange(states)
            done = last and rng.random() < 0.6
            truncated = last and not done
            next_action = rng.randrange(actions) if (sarsa and not last) else None
            out.append((state, action, rng.uniform(-1, 1), next_state, done, truncated, next_action))
            if last:
                break
            state, action = next_state, next_action if next_action is not None else rng.randrange(actions)
    return out


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
@pytest.mark.parametrize("n_step", [1, 2, 5])
def test_the_rust_update_rule_equals_the_textbook_one_exactly(algorithm, n_step):
    rng = random.Random(n_step * 10 + len(algorithm))
    bits, actions = 4, 3
    transitions = _episodes(rng, 1 << bits, actions, count=60, sarsa=algorithm == "sarsa")
    params = {"alpha": 0.3, "gamma": 0.9, "n_step": float(n_step), "initial_q": 0.25}
    rust = _native.tabular_replay(algorithm, bits, actions, params, transitions)
    python = replay(
        algorithm == "sarsa", 1 << bits, actions, transitions, alpha=0.3, gamma=0.9, n_step=n_step, initial_q=0.25
    )
    assert rust == python


def test_q_learning_learns_snake_through_the_adapter():
    trainer = rl.Trainer("q_learning", FEATURES, seed=1, params={"epsilon_decay_steps": 100_000})
    before = statistics.fmean(e["score"] for e in trainer.evaluate(list(range(20_000, 20_030)), 1000))
    stats = trainer.train(300_000)
    after = statistics.fmean(e["score"] for e in trainer.evaluate(list(range(20_000, 20_030)), 1000))
    assert after > before + 8 and after > 12  # greedy is ~18 on the leaderboard; a random policy scores ~0.2
    extras = stats["extras"]
    assert extras["epsilon"] == 0.05 and 100 < extras["states_visited"] <= 2048
    assert 0 < stats["entropy"] < 0.3  # epsilon 0.05 over 3 actions


def test_the_snapshot_is_a_qtable_champion():
    trainer = rl.Trainer("sarsa", FEATURES, seed=0, params={"n_step": 2})
    trainer.train(5_000)
    snapshot = json.loads(trainer.snapshot())
    assert snapshot["type"] == "qtable" and snapshot["algorithm"] == "sarsa"
    assert snapshot["discretizer"] == {"kind": "binary", "bits": 11}
    assert snapshot["actions"] == {"kind": "discrete", "count": 3}
    assert len(snapshot["values"]) == 2048 * 3


def test_reach1d_gets_binned_states_and_actions():
    trainer = rl.Trainer("q_learning", "reach1d", seed=0, params={"action_bins": 5}, max_episode_steps=200)
    trainer.train(20_000)
    snapshot = json.loads(trainer.snapshot())
    assert snapshot["discretizer"]["kind"] == "bins" and len(snapshot["values"]) == 325 * 5
    assert snapshot["actions"] == {"kind": "continuous", "values": [-1.0, -0.5, 0.0, 0.5, 1.0]}


def test_tabular_refuses_what_a_table_cannot_hold_and_unknown_params():
    with pytest.raises(ValueError, match="tabular"):
        rl.Trainer("q_learning", "snake/grid-flat.v1+relative3.v1", seed=0)
    with pytest.raises(ValueError, match="no parameter"):
        rl.Trainer("q_learning", FEATURES, seed=0, params={"learning_rate": 0.1})
    with pytest.raises(ValueError, match="not a tabular"):
        _native.tabular_replay("dqn", 2, 2, {}, [])
