import checkers_neuro_run
import run_context
from evolve import WeightVector
from telemetry import FileMetricsStore, SqliteRunRegistry


def test_training_run_evolves_a_position_evaluator_through_match_fitness(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(checkers_neuro_run, "MONITOR_GAMES", 2)

    run_id = checkers_neuro_run.main(
        generations=2, population_size=6, hidden=4, opponents=("random", "material-1"), held_out_every=1
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
    a = checkers_neuro_run.monitor_score(genome, ["random"], games=4)
    assert a == checkers_neuro_run.monitor_score(genome, ["random"], games=4)
