import pytest

from arena import snake
from arena.seeding import TRAINING_POOL, SeedStrategy


def test_fixed_strategy_is_the_original_benchmark_seeds():
    seeds = SeedStrategy.parse("fixed:5")

    assert seeds.initial() == [0, 1, 2, 3, 4]
    assert not seeds.resamples
    assert seeds.config() == {"seed_strategy": "fixed:5", "training_seeds": [0, 1, 2, 3, 4]}


def test_resample_draws_fresh_reproducible_seeds_outside_every_evaluation_set():
    a, b = SeedStrategy.parse("resample:5", rng_seed=1), SeedStrategy.parse("resample:5", rng_seed=1)
    draws = [a.draw() for _ in range(3)]

    assert draws == [b.draw() for _ in range(3)]  # reproducible from the rng seed
    assert draws[0] != draws[1]  # fresh every generation
    evaluation = set(snake.HELD_OUT_SEEDS) | set(snake.MONITOR_SEEDS)
    for draw in draws:
        assert len(draw) == 5 and all(seed in TRAINING_POOL and seed not in evaluation for seed in draw)
    assert a.config()["training_seeds"] is None


@pytest.mark.parametrize("bad", ["fixed", "resample:x", "bogus:3", "fixed:0"])
def test_bad_strategies_are_rejected(bad):
    with pytest.raises(ValueError):
        SeedStrategy.parse(bad)
