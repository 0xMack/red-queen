"""The Rust game core against its oracle (reference_snake.py): identical games, step for step, for the
same seed -- observations under both observers, rewards, termination, score, render order -- across
many seeds, board sizes and action sequences (random and greedy, which reach long, food-eating games)."""

import random

import pytest
from games import _native, baselines
from games.snake import Snake, SnakeFeatures, SnakeGridFlat

from reference_snake import Pcg32, ReferenceSnake, reference_greedy


@pytest.mark.parametrize("seed", [0, 1, 42, 2**31 - 1, 2**40 + 7, 2**64 - 1])
def test_pcg32_matches_the_python_reference(seed):
    native, reference = _native.Pcg32(seed), Pcg32(seed)
    assert [native.next_u32() for _ in range(100)] == [reference.next_u32() for _ in range(100)]
    for n in (1, 2, 3, 7, 100, 2**32 - 1):
        assert [native.bounded(n) for _ in range(50)] == [reference.bounded(n) for _ in range(50)]


def test_pcg32_reference_vector():
    # pcg32-demo, pcg32_srandom_r(42, 54): the published first outputs.
    rng = Pcg32(42)
    assert [rng.next_u32() for _ in range(3)] == [0xA15C02B7, 0x7B47F409, 0xBA1D3330]


def _play_both(seed: int, width: int, height: int, choose) -> int:
    native = Snake(width=width, height=height, seed=seed, observer=SnakeFeatures())
    grid = Snake(width=width, height=height, seed=seed, observer=SnakeGridFlat())
    reference = ReferenceSnake(width=width, height=height, seed=seed)
    observation = native.reset()
    assert grid.reset() == reference.grid_flat()
    assert observation == reference.features()
    for step in range(2000):
        action = choose(observation, step)
        observation, reward, done = native.step(action)
        grid_observation, _, _ = grid.step(action)
        expected_reward, expected_done = reference.step(action)
        assert (reward, done) == (expected_reward, expected_done), f"seed {seed} step {step}"
        if not reference.alive:
            assert not native.alive
            break
        assert observation == reference.features(), f"seed {seed} step {step}"
        assert grid_observation == reference.grid_flat(), f"seed {seed} step {step}"
        assert native.body == reference.body and native.food == reference.food
        assert list(native.render_state()["cells"].items()) == [((x, y), label) for x, y, label in reference.cells()]
    assert native.score == reference.score
    return reference.score


@pytest.mark.parametrize("seed", range(40))
def test_random_play_is_identical(seed):
    rng = random.Random(seed)
    _play_both(seed, 10, 10, lambda _obs, _step: rng.choice((-1, 0, 1)))


@pytest.mark.parametrize("seed", range(40))
def test_greedy_play_is_identical(seed):
    # Greedy eats a lot (mean ~18 food), so these games exercise growth, food placement on a crowded
    # board, and starvation.
    _play_both(seed, 10, 10, lambda obs, _step: reference_greedy(obs))


@pytest.mark.parametrize(("width", "height"), [(5, 5), (7, 12), (20, 6), (4, 3)])
def test_other_board_sizes(width, height):
    for seed in range(10):
        _play_both(seed, width, height, lambda obs, _step: reference_greedy(obs))


def test_baselines_match_their_python_originals():
    greedy = baselines.get("snake", "greedy").factory(0)
    rng = random.Random(0)
    for _ in range(2000):
        observation = [float(rng.random() < 0.3) for _ in range(3)] + [0.0] * 4 + [float(rng.random() < 0.5) for _ in range(4)]
        observation[3 + rng.randrange(4)] = 1.0
        assert greedy(observation) == reference_greedy(observation)
    random_policy = baselines.get("snake", "random").factory(5)
    reference = Pcg32(5)
    assert [random_policy([0.0] * 11) for _ in range(200)] == [reference.bounded(3) - 1 for _ in range(200)]


def test_a_board_too_narrow_for_the_starting_snake_is_rejected():
    with pytest.raises(ValueError, match="starting snake"):
        Snake(width=3, height=5)
