"""The Rust game core against its oracle (reference_snake.py): identical games, step for step, for the
same seed -- observations under both observers, rewards, termination, score, render order -- across
many seeds, board sizes and action sequences (random and greedy, which reach long, food-eating games)."""

import random

import pytest
from reference_snake import Pcg32, ReferenceSnake, reference_greedy

from games import _native, baselines
from games.snake import Snake, SnakeEgocentric, SnakeFeatures, SnakeGridFlat


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
    ego = Snake(width=width, height=height, seed=seed, observer=SnakeEgocentric())
    reference = ReferenceSnake(width=width, height=height, seed=seed)
    observation = native.reset()
    assert grid.reset() == reference.grid_flat()
    assert ego.reset() == reference.egocentric()
    assert observation == reference.features()
    for step in range(2000):
        action = choose(observation, step)
        observation, reward, done = native.step(action)
        grid_observation, _, _ = grid.step(action)
        ego_observation, _, _ = ego.step(action)
        expected_reward, expected_done = reference.step(action)
        assert (reward, done) == (expected_reward, expected_done), f"seed {seed} step {step}"
        if not reference.alive:
            assert not native.alive
            break
        assert observation == reference.features(), f"seed {seed} step {step}"
        assert grid_observation == reference.grid_flat(), f"seed {seed} step {step}"
        assert ego_observation == reference.egocentric(), f"seed {seed} step {step}"
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


# --- Checkers ------------------------------------------------------------------------------------


def _play_checkers(seed: int, max_moves_without_capture: int = 40) -> None:
    from reference_checkers import Checkers as ReferenceCheckers

    from games.checkers import Checkers

    native, reference = Checkers(max_moves_without_capture), ReferenceCheckers(max_moves_without_capture)
    rng = random.Random(seed)
    assert native.reset() == reference.reset()
    for turn in range(400):
        moves = native.legal_moves()
        assert moves == reference.legal_moves(), f"seed {seed} turn {turn}: move lists (incl. order) differ"
        assert list(native.board.items()) == list(reference.board.items())
        assert native.render_state() == reference.render_state()
        for move in moves[:5]:
            assert native.simulate(move) == reference.simulate(move)
        # Mix random play with a capture-hungry preference, so games reach kings and long chains.
        move = max(moves, key=len) if rng.random() < 0.5 else rng.choice(moves)
        result = native.step(move)
        assert result == reference.step(move), f"seed {seed} turn {turn}"
        assert native.winner() == reference.winner() and native.current_player() == reference.current_player()
        if result[2]:
            break


@pytest.mark.parametrize("seed", range(60))
def test_checkers_games_are_identical(seed):
    _play_checkers(seed)


@pytest.mark.parametrize("seed", range(10))
def test_checkers_draw_limit_is_identical(seed):
    _play_checkers(seed, max_moves_without_capture=6)


def test_checkers_positions_set_from_python_play_identically():
    from reference_checkers import Checkers as ReferenceCheckers

    from games.checkers import Checkers

    board = {(4, 4): (0, True), (5, 3): (1, False), (3, 5): (1, False), (1, 5): (1, True), (0, 2): (0, False)}
    native, reference = Checkers(), ReferenceCheckers()
    native.board = reference.board = board
    native.current_player_index = reference.current_player_index = 0
    assert native.legal_moves() == reference.legal_moves()
    move = native.legal_moves()[0]
    assert native.step(move) == reference.step(move)
    assert list(native.board.items()) == list(reference.board.items())


# --- ReachTarget1D --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_reach1d_is_bit_identical(seed):
    from reference_reach1d import ReachTarget1D as ReferenceReach

    from games.reach1d import ReachTarget1D

    rng = random.Random(seed)
    kwargs = {"target": rng.uniform(-9, 9), "start_position": rng.uniform(-9, 9), "start_velocity": rng.uniform(-3, 3)}
    native, reference = ReachTarget1D(**kwargs), ReferenceReach(**kwargs)
    assert native.reset() == reference.reset()
    for _ in range(500):
        action = rng.choice([rng.uniform(-3, 3), 1.0, -1.0, 0, float("nan"), float("inf")])
        assert native.step(action) == reference.step(action) or _nan_equal(native, reference)
        assert (native.position, native.velocity) == (reference.position, reference.velocity) or _nan_equal(native, reference)


def _nan_equal(native, reference) -> bool:
    import math

    return all(math.isnan(a) == math.isnan(b) for a, b in ((native.position, reference.position), (native.velocity, reference.velocity)))


def test_games_deep_copy_into_independent_states():
    import copy

    from games.checkers import Checkers
    from games.reach1d import ReachTarget1D

    checkers = Checkers()
    child = copy.deepcopy(checkers)
    child.step(child.legal_moves()[0])
    assert checkers.current_player() == 0 and child.current_player() == 1

    snake = Snake(seed=3)
    twin = copy.deepcopy(snake)
    for _ in range(2):  # the start is 5 cells from the right wall
        assert snake.step(0) == twin.step(0)  # same PRNG state, same game
    twin.step(1)
    assert snake.body != twin.body

    reach = ReachTarget1D()
    other = copy.copy(reach)
    other.step(1.0)
    assert reach.position == 0.0 != other.position
