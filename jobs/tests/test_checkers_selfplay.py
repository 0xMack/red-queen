import checkers_selfplay_run
import evaluate_versus
import run_context
from evolve import WeightVector
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry


def test_a_self_play_run_is_recorded_and_joins_the_versus_leaderboard(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(checkers_selfplay_run, "MONITOR_GAMES", 2)

    run_id = checkers_selfplay_run.main(
        iterations=evaluate_versus.MIN_GENERATIONS, games_per_iteration=20, depth=1, held_out_every=5, rng_seed=3
    )

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


def test_an_alphazero_run_is_recorded_and_its_value_head_joins_the_leaderboard(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(checkers_selfplay_run, "MONITOR_GAMES", 2)
    monkeypatch.setattr(checkers_selfplay_run, "MCTS_GAMES", 2)
    params = {"hidden": 8, "hidden_layers": 1, "simulations": 4}

    run_id = checkers_selfplay_run.main(
        iterations=evaluate_versus.MIN_GENERATIONS,
        games_per_iteration=2,
        depth=1,
        held_out_every=5,
        rng_seed=3,
        params=params,
        algorithm="alphazero",
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

    # and a saved value network (here this run's own) starts an AlphaZero run: the trunk and value unit, a uniform policy
    child = checkers_selfplay_run.main(
        iterations=1, games_per_iteration=1, depth=1, rng_seed=4, params=params, algorithm="alphazero", init_run=run_id
    )
    assert registry.get_run(child).config["init_run"] == run_id


def test_the_recipe_trains_td_then_fine_tunes_it(tmp_path, monkeypatch):
    import checkers_recipe_run

    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(checkers_selfplay_run, "MONITOR_GAMES", 2)
    td, leaf = checkers_recipe_run.main(
        hidden=8, layers=1, td_games=40, leaf_games=4, leaf_depth=2, depth=1, games_per_iteration=20
    )
    registry = SqliteRunRegistry(tmp_path / "runs.db")
    child = registry.get_run(leaf).config
    assert child["init_run"] == td and child["params"]["search_depth"] == 2 and child["layer_sizes"] == [32, 8, 1]
    assert "experiment" not in child  # recipe runs are leaderboard entrants
    # an existing TD run is fine-tuned again (another depth's entrant) without retraining it
    assert checkers_recipe_run.main(hidden=8, layers=1, leaf_games=4, leaf_depth=2, depth=2, td_run=td)[0] == td
