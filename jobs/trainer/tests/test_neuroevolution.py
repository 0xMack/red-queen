import arena.snake
from evolve import WeightVector
from telemetry import FileMetricsStore, SqliteRunRegistry

import trainer.snake
from trainer import checkers, train


def test_a_snake_run_records_its_seed_strategy_and_held_out_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trainer.snake,
        "monitor_score",
        lambda interface, policy: arena.snake.monitor_score(interface, policy, (20_000, 20_001)),
    )

    train(
        {
            "game": "snake",
            "algorithm": "neuroevolution",
            "budget": {"generations": 3},
            "held_out_every": 2,
            "params": {"population_size": 6, "seeds": "resample:2"},
        }
    )

    run = SqliteRunRegistry(tmp_path / "runs.db").list_runs()[0]
    history = FileMetricsStore(tmp_path / "metrics").history(run.run_id)
    assert run.config["seed_strategy"] == "resample:2" and run.config["training_seeds"] is None
    # every 2nd generation plus the last one
    assert [h.held_out_score is not None for h in history] == [True, False, True]
    assert run.summary["held_out_score"] == history[-1].held_out_score
    assert run.summary["cost"]["episodes"] == 6 * 3 * 2  # genomes x generations x seeds


def test_a_checkers_run_evolves_a_position_evaluator_through_match_fitness(tmp_path):
    run_id = train(
        {
            "game": "checkers",
            "algorithm": "neuroevolution",
            "budget": {"generations": 2},
            "held_out_every": 1,
            "params": {"population_size": 6, "hidden": 4, "opponents": ["random", "material-1"], "monitor_games": 2},
        }
    )

    run = SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id)
    assert run.status == "completed"
    assert run.config["game"] == "checkers" and run.config["opponents"] == ["random", "material-1"]
    history = FileMetricsStore(tmp_path / "metrics").history(run_id)
    assert [g.generation for g in history] == [0, 1]
    assert all(g.held_out_score is not None and -1.0 <= g.held_out_score <= 1.0 for g in history)
    # 2 opponents x both seats = 4 fitness cases, each in [-1, 1]
    assert -1.0 <= history[-1].best_fitness <= 1.0


def test_a_champion_scored_against_the_pool_is_deterministic():
    genome = WeightVector(weights=tuple([0.1] * (32 * 3 + 3 + 3 + 1)), layer_sizes=(32, 3, 1))
    a = checkers.monitor_score(genome, ["random"], games=4)
    assert a == checkers.monitor_score(genome, ["random"], games=4)
