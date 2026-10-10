"""The bandit jobs (docs/design/0011): evolving a strategy's settings records an ordinary run, and the leaderboard
evaluation enters its champion beside the hand-set strategies."""

import bandit_evolve_run
import evaluate_bandit
from evolve import WeightVector
from jobcore import open_sink
from telemetry import SqliteEvaluationStore


def test_decoded_settings_stay_in_range():
    for g in (-50.0, 0.0, 50.0):
        p = bandit_evolve_run.decode(WeightVector((g, g, g, g), (4,)))
        assert 0 <= p["epsilon"] <= 1 and 20 <= p["decay"] <= 400 and 0 <= p["alpha"] <= 0.5 and 0 <= p["initial"] <= 2


def test_an_evolved_strategy_is_recorded_and_ranked(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit_evolve_run, "POPULATION", 6)
    monkeypatch.setattr(bandit_evolve_run, "GAMES_PER_SCENARIO", 10)
    monkeypatch.setattr(evaluate_bandit, "HELD_OUT", 20)

    run_id = bandit_evolve_run.main(["classic"], generations=3)
    stores = open_sink()
    run = stores.registry.get_run(run_id)
    assert run.status == "completed" and run.config["representation"] == "evolved_bandit"
    history = stores.metrics.history(run_id)
    assert len(history) == 3 and history[0].held_out_score is not None and history[-1].held_out_score is not None
    assert set(history[-1].extras) == {"epsilon", "decay", "alpha", "initial"}

    evaluate_bandit.main()
    records = {r.entrant_id: r for r in SqliteEvaluationStore(tmp_path / "evaluations.db").list("bandit")}
    evolved = records[f"run:{run_id}"]
    assert evolved.entrant_kind == "champion" and evolved.run_id == run_id
    assert evolved.metrics["bandit"]["strategy"] == "epsilon_greedy"
    assert set(evolved.metrics["bandit"]["scenarios"]) >= {"classic", "detour", "two-lamps:lamp.v1"}
    assert "baseline:random" in records and "strategy:thompson" in records
    # Evolution's cost is the evolved entrant's training; a hand-set strategy trains nothing.
    assert evolved.metrics["training"]["generations"] == 3 and not evolved.metrics["training"].get("none")
    assert records["strategy:thompson"].metrics["training"]["none"]
