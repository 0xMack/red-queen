"""Checkers self-play runs (the trainer's `td_lambda` / `alphazero`) are recorded like any other Checkers run and join
the versus leaderboard (jobs/evaluate_versus.py); the recipe chains two of them."""

import evaluate_versus
from evolve import WeightVector
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry
from trainer import train


def selfplay(algorithm: str, iterations: int, games: int, agent: dict | None = None, **fields) -> str:
    extra = fields.pop("params", {})
    return train(
        {
            "game": "checkers",
            "algorithm": algorithm,
            "budget": {"iterations": iterations},
            "params": {"games_per_iteration": games, "depth": 1, "monitor_games": 2, "agent": agent or {}, **extra},
            **fields,
        }
    )


def test_a_self_play_run_is_recorded_and_joins_the_versus_leaderboard(tmp_path):
    run_id = selfplay("td_lambda", evaluate_versus.MIN_GENERATIONS, 20, held_out_every=5, seed=3)

    registry, metrics = SqliteRunRegistry(tmp_path / "runs.db"), FileMetricsStore(tmp_path / "metrics")
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    run = registry.get_run(run_id)
    assert run.status == "completed" and run.config["representation"] == "td_lambda"
    assert run.config["interface"] == evaluate_versus.INTERFACE and run.config["layer_sizes"] == [32, 16, 1]
    history = metrics.history(run_id)
    assert len(history) == evaluate_versus.MIN_GENERATIONS
    assert [h.held_out_score is not None for h in history][:6] == [True, False, False, False, False, True]
    assert history[-1].extras["games"] == 20 * evaluate_versus.MIN_GENERATIONS
    assert -1 <= history[-1].mean_fitness <= 1 and history[-1].extras["td_loss"] >= 0
    champion = WeightVector.from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    assert champion.layer_sizes == (32, 16, 1)
    assert run.summary["cost"]["episodes"] == 20 * evaluate_versus.MIN_GENERATIONS

    (entrant,) = evaluate_versus.champion_entrants(registry, metrics, artifacts)
    assert entrant["label"].startswith("TD(λ) self-play 32 → 16 → 1") and "self-play" in entrant["model"]


def test_an_alphazero_run_is_recorded_and_its_value_head_joins_the_leaderboard(tmp_path):
    agent = {"hidden": 8, "hidden_layers": 1, "simulations": 4}
    run_id = selfplay(
        "alphazero", evaluate_versus.MIN_GENERATIONS, 2, agent, held_out_every=5, seed=3, params={"mcts_games": 2}
    )

    registry, metrics = SqliteRunRegistry(tmp_path / "runs.db"), FileMetricsStore(tmp_path / "metrics")
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    run = registry.get_run(run_id)
    assert run.status == "completed" and run.config["representation"] == "alphazero"
    assert run.config["layer_sizes"] == [32, 8, 1] and "AlphaZero" in run.config["training"]
    history = metrics.history(run_id)
    assert {"value_loss", "policy_loss", "search_shift", "policy_entropy_start"} <= set(history[-1].extras)
    assert 0 <= history[0].extras["mcts_points"] <= 1 and "mcts_points" not in history[1].extras
    champion = WeightVector.from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    assert champion.layer_sizes == (32, 8, 1)
    full = artifacts.get_program(f"{history[-1].champion_ref}-net").decode("utf-8")
    assert '"layer_sizes": [32, 8, 129]' in full and '"heads": "value1+policy128.v1"' in full

    (entrant,) = evaluate_versus.champion_entrants(registry, metrics, artifacts)
    assert entrant["label"].startswith("AlphaZero self-play 32 → 8 → 1")
    assert "trained by AlphaZero self-play" in entrant["model"]

    # and a saved value network (here this run's own, by an id prefix) starts an AlphaZero run: the trunk and value
    # unit, a uniform policy
    child = selfplay("alphazero", 1, 1, agent, seed=4, init_from={"run": run_id[:8]}, params={"mcts_games": 2})
    assert registry.get_run(child).config["init_run"] == run_id


def test_init_from_an_experiments_arm_continues_that_seeds_run(tmp_path):
    agent = {"hidden": 8, "hidden_layers": 1}
    parent = selfplay("td_lambda", 1, 2, agent, seed=5, tags={"experiment": "e", "arm": "a"})
    selfplay("td_lambda", 1, 2, agent, seed=6, tags={"experiment": "e", "arm": "a"})
    child = selfplay("td_lambda", 1, 2, agent, seed=5, init_from={"experiment": "e", "arm": "a"})
    assert SqliteRunRegistry(tmp_path / "runs.db").get_run(child).config["init_run"] == parent


def test_the_recipe_trains_td_then_fine_tunes_it(tmp_path):
    import checkers_recipe_run

    td, leaf = checkers_recipe_run.main(
        hidden=8, layers=1, td_games=40, leaf_games=4, leaf_depth=2, depth=1, games_per_iteration=20, monitor_games=2
    )
    registry = SqliteRunRegistry(tmp_path / "runs.db")
    child = registry.get_run(leaf).config
    assert child["init_run"] == td and child["params"]["search_depth"] == 2 and child["layer_sizes"] == [32, 8, 1]
    assert "experiment" not in child  # recipe runs are leaderboard entrants
    # an existing TD run is fine-tuned again (another depth's entrant) without retraining it
    again = checkers_recipe_run.main(
        hidden=8, layers=1, leaf_games=4, leaf_depth=2, depth=2, td_run=td, monitor_games=2
    )
    assert again[0] == td
