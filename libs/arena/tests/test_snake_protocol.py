from games import baselines, interfaces
from games.snake import BENCHMARK_SEEDS

from arena import snake
from arena.seeding import TRAINING_POOL

FEATURES = interfaces.get("snake/features.v1+relative3.v1")


def test_score_stats_reports_spread_not_just_the_mean():
    stats = snake.score_stats([0, 2, 4, 10])

    assert stats["mean"] == 4.0
    assert stats["median"] == 3.0
    assert (stats["min"], stats["max"]) == (0, 10)
    assert stats["zero_rate"] == 0.25
    assert stats["ci95"] > 0


def test_greedy_baseline_beats_random_on_held_out_seeds(monkeypatch):
    monkeypatch.setattr(snake, "HELD_OUT_SEEDS", tuple(range(10_000, 10_020)))

    greedy = snake.measure_quality(FEATURES, baselines.get("snake", "greedy").factory, training_seeds=[0])
    random_ = snake.measure_quality(FEATURES, baselines.get("snake", "random").factory, training_seeds=[0])

    assert greedy["mean"] > random_["mean"] + 5
    assert greedy["n"] == 20
    assert abs(greedy["generalization_gap"] - (greedy["train_mean"] - greedy["mean"])) < 1e-3


def test_evaluation_seeds_are_disjoint_from_every_training_seed():
    held_out, monitor = set(snake.HELD_OUT_SEEDS), set(snake.MONITOR_SEEDS)
    assert not held_out & monitor
    for training in (set(BENCHMARK_SEEDS), set(TRAINING_POOL)):
        assert not training & (held_out | monitor)
