"""The game adapters and the training loop.

An adapter is right when the numbers are: a games-crate baseline played through the Snake adapter must score
exactly what `jobs/evaluate.py`'s protocol records for the same seeds.
"""

import math

import pytest
from games import baselines, interfaces

import rl

FEATURES = "snake/features.v1+relative3.v1"
BOARD = {"width": 10, "height": 10}
SEEDS = list(range(10_000, 10_060))  # the first 60 of the leaderboard's held-out games
MAX_STEPS = 1000  # the protocol's cap


def _python_protocol(policy_factory, seed: int) -> tuple[int, int]:
    """jobs/evaluate.py's play_episode, inlined (jobs/ isn't importable from libs/)."""
    game = interfaces.get(FEATURES).make_game(seed=seed, **BOARD)
    observation = game.reset()
    policy = policy_factory(seed)
    steps = 0
    for _ in range(MAX_STEPS):
        observation, _reward, done = game.step(policy(observation))
        steps += 1
        if done:
            break
    return game.score, steps


@pytest.mark.parametrize(("native", "python"), [("snake-random", "random"), ("snake-greedy", "greedy")])
def test_baselines_through_the_adapter_score_what_the_protocol_records(native, python):
    factory = baselines.get("snake", python).factory
    episodes = rl.evaluate_baseline(native, FEATURES, SEEDS, MAX_STEPS)
    assert [(e["score"], e["steps"]) for e in episodes] == [_python_protocol(factory, s) for s in SEEDS]


def test_env_info_describes_each_environment():
    assert rl.env_info(FEATURES) == (11, [3.0], "discrete")
    assert rl.env_info("snake/egocentric.v1+relative3.v1")[0] == 27
    assert rl.env_info("snake/grid-flat.v1+relative3.v1", 12, 8)[0] == 96
    assert rl.env_info("reach1d") == (2, [-1.0, 1.0], "continuous")
    with pytest.raises(ValueError, match="unknown environment"):
        rl.env_info("chess")


def test_a_random_trainer_reports_iterations_and_totals():
    trainer = rl.Trainer("random", FEATURES, seed=5, max_episode_steps=1000)
    first = trainer.train(3000)
    second = trainer.train(3000)
    assert first["steps"] == 3000 and second["total_steps"] == 6000
    assert first["episodes"] and all(100_000 <= e["seed"] < 1_000_000 for e in first["episodes"])
    assert second["total_episodes"] == len(first["episodes"]) + len(second["episodes"])
    assert math.isclose(first["entropy"], math.log(3))
    assert first["extras"] == {}
    assert trainer.snapshot() == '{"type": "random", "num_actions": 3}'
    assert (trainer.algorithm, trainer.env_id) == ("random", FEATURES)


def test_same_seed_same_run_and_evaluation_leaves_training_untouched():
    a = rl.Trainer("random", FEATURES, seed=9)
    b = rl.Trainer("random", FEATURES, seed=9)
    a.train(1500)
    b.train(1500)
    b.evaluate([20_000, 20_001], 500)  # a fresh environment and stream: training carries on as if it never ran
    assert a.train(2500) == b.train(2500)
    assert rl.Trainer("random", FEATURES, seed=10).train(2500) != rl.Trainer("random", FEATURES, seed=9).train(2500)


def test_evaluation_is_per_game_deterministic():
    trainer = rl.Trainer("random", FEATURES, seed=1)
    once = trainer.evaluate([20_000, 20_001, 20_002], 1000)
    trainer.train(500)
    assert trainer.evaluate([20_000, 20_001, 20_002], 1000) == once  # the random agent draws per game seed


def test_reach1d_runs_to_the_step_cap_with_continuous_actions():
    trainer = rl.Trainer("random", "reach1d", seed=0, max_episode_steps=200)
    stats = trainer.train(1000)
    assert [e["steps"] for e in stats["episodes"]] == [200] * 5
    assert all(e["score"] <= 0 for e in stats["episodes"])
    assert math.isclose(stats["entropy"], math.log(2.0))  # U(-1, 1)


def test_bad_configurations_are_errors():
    with pytest.raises(ValueError, match="unknown algorithm"):
        rl.Trainer("sarsa-lambda-9000", FEATURES, seed=0)
    with pytest.raises(ValueError, match="no parameter"):
        rl.Trainer("random", FEATURES, seed=0, params={"epsilon": 0.1})
    with pytest.raises(ValueError, match="seed pool"):
        rl.Trainer("random", FEATURES, seed=0, seed_pool=(5, 5))
    with pytest.raises(ValueError, match="board"):
        rl.Trainer("random", FEATURES, seed=0, width=3)
