import random

import pytest
from reference_checkers_strategies import (
    evaluator_scores,
    material_1_scores,
    material_2_scores,
)

from games import _native
from games.checkers import Checkers
from games.checkers_strategies import (
    STRATEGIES,
    evaluator,
    first_legal,
    material_2,
    random_strategy,
)

LAYERS = (32, 5, 1)


def _weights(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-0.6, 0.6) for _ in range(5 * 33 + 6)]


def _positions(games: int = 6, seed: int = 0):
    """Real mid-game positions: random self-play, every position along the way (including captures,
    multi-jumps and kings once the game gets there)."""
    rng = random.Random(seed)
    for _ in range(games):
        env = Checkers()
        env.reset()
        done = False
        for _ in range(120):
            if done:
                break
            yield env
            _, _, done = env.step(rng.choice(env.legal_moves()))


def _acceptable(scores: list[float]) -> set[int]:
    top = max(scores)
    return {i for i, s in enumerate(scores) if s == top}


def _pick(env: Checkers, name: str, seed: int, **network) -> int:
    strategy = _native.CheckersStrategy(name, seed, **network)
    return strategy.pick(env._core)


def test_first_legal_is_the_first_legal_move():
    env = Checkers()
    env.reset()
    assert first_legal(env, random.Random(0))(None, env.legal_moves()) == env.legal_moves()[0]


def test_every_strategy_returns_a_legal_move_from_the_opening():
    env = Checkers()
    env.reset()
    for name, factory in STRATEGIES.items():
        assert factory(env, random.Random(0))(None, env.legal_moves()) in env.legal_moves(), name


def test_material_1_always_picks_a_top_scoring_move():
    for i, env in enumerate(_positions()):
        assert _pick(env, "material-1", i) in _acceptable(material_1_scores(env))


def test_material_2_always_picks_a_top_scoring_move():
    for i, env in enumerate(_positions(games=3)):
        assert _pick(env, "material-2", i) in _acceptable(material_2_scores(env))


def test_evaluator_always_picks_a_top_scoring_move_under_the_reference_network():
    weights = _weights(1)
    for i, env in enumerate(_positions()):
        picked = _pick(env, "evaluator", i, weights=weights, layer_sizes=list(LAYERS))
        # Float-exact tie-handling across two tanh implementations would be flaky; near-ties are fine.
        scores = evaluator_scores(env, weights, LAYERS)
        assert scores[picked] >= max(scores) - 1e-9


def test_random_is_seeded_and_covers_the_moves():
    env = Checkers()
    env.reset()
    picks = lambda seed: [_pick(env, "random", seed + n) for n in range(60)]
    assert picks(5) == picks(5)
    assert len(set(picks(5))) == len(env.legal_moves())  # 7 opening moves, all reachable


def test_a_strategy_is_reproducible_from_its_rng():
    def game(seed):
        env = Checkers()
        rng = random.Random(seed)
        strategies = {0: material_2(env, rng), 1: random_strategy(env, rng)}
        env.reset()
        for _ in range(60):
            if env.step(strategies[env.current_player()](None, env.legal_moves()))[2]:
                break
        return env.board

    assert game(3) == game(3)


def test_evaluator_factory_plays_and_rejects_a_wrong_shaped_network():
    env = Checkers()
    env.reset()
    move = evaluator(_weights(2), LAYERS)(env, random.Random(0))(None, env.legal_moves())
    assert move in env.legal_moves()
    with pytest.raises(ValueError):
        _native.CheckersStrategy("evaluator", 0, weights=[0.0] * 10, layer_sizes=[32, 1])
    with pytest.raises(ValueError):
        _native.CheckersStrategy("evaluator", 0)
    with pytest.raises(ValueError):
        _native.CheckersStrategy("nonsense", 0)


def test_two_ply_lookahead_beats_random_from_both_seats():
    def winner(a, b, seed):
        env = Checkers()
        rng = random.Random(seed)
        strategies = {0: a(env, rng), 1: b(env, rng)}
        env.reset()
        for _ in range(300):
            if env.step(strategies[env.current_player()](None, env.legal_moves()))[2]:
                break
        return env.winner()

    wins = sum(winner(material_2, random_strategy, s) == 0 for s in range(30))
    wins += sum(winner(random_strategy, material_2, s) == 1 for s in range(30))
    assert wins >= 48  # 80% of 60 (never a loss; the rest are draws) -- equal-strength play would win ~35%
