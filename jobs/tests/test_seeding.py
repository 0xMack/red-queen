import evaluate
import pytest
import run_context
import snake_neuro_run
from seeding import TRAINING_POOL, SeedStrategy
from telemetry import FileMetricsStore, SqliteRunRegistry


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
    evaluation = set(evaluate.HELD_OUT_SEEDS) | set(evaluate.MONITOR_SEEDS)
    for draw in draws:
        assert len(draw) == 5 and all(seed in TRAINING_POOL and seed not in evaluation for seed in draw)
    assert a.config()["training_seeds"] is None


@pytest.mark.parametrize("bad", ["fixed", "resample:x", "bogus:3", "fixed:0"])
def test_bad_strategies_are_rejected(bad):
    with pytest.raises(ValueError):
        SeedStrategy.parse(bad)


def test_training_run_records_strategy_and_held_out_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(snake_neuro_run, "POPULATION_SIZE", 6)
    monkeypatch.setattr(evaluate, "MONITOR_SEEDS", (20_000, 20_001))
    monkeypatch.setattr(snake_neuro_run, "monitor_score", lambda interface, policy: evaluate.monitor_score(interface, policy, (20_000, 20_001)))

    snake_neuro_run.main(seed_strategy="resample:2", held_out_every=2, generations=3)

    run = SqliteRunRegistry(tmp_path / "runs.db").list_runs()[0]
    history = FileMetricsStore(tmp_path / "metrics").history(run.run_id)
    assert run.config["seed_strategy"] == "resample:2" and run.config["training_seeds"] is None
    # every 2nd generation plus the last one
    assert [h.held_out_score is not None for h in history] == [True, False, True]
    assert run.summary["held_out_score"] == history[-1].held_out_score
    assert run.summary["cost"]["episodes"] == 6 * 3 * 2  # genomes x generations x seeds
