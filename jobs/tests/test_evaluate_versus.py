import evaluate_versus
from evolve import WeightVector
from games.checkers_strategies import evaluator


def _entrants():
    weights = tuple([0.05] * (32 * 4 + 4 + 4 + 1))
    champion = WeightVector(weights=weights, layer_sizes=(32, 4, 1))
    trained = {
        "entrant_id": "run:fake",
        "label": "Neuroevolution 32 → 4 → 1",
        "factory": evaluator(champion.weights, champion.layer_sizes),
        "model": "MLP",
        "parameters": len(weights),
        "artifact_bytes": 1,
    }
    return [*evaluate_versus.baseline_entrants(), trained]


def test_round_robin_is_zero_sum_and_symmetric():
    entrants = _entrants()
    results = evaluate_versus.round_robin(entrants, games=4)

    for a in entrants:
        assert set(results[a["entrant_id"]]) == {e["entrant_id"] for e in entrants if e is not a}
    # Every game hands out exactly one point, and one side's points are the other's mirror image.
    total = sum(p for by in results.values() for games in by.values() for p, _ in games)
    pairs = len(entrants) * (len(entrants) - 1) // 2
    assert total == pairs * 4 * 1.0 * 1.0
    a, b = entrants[0]["entrant_id"], entrants[1]["entrant_id"]
    assert [p for p, _ in results[a][b]] == [1.0 - p for p, _ in results[b][a]]


def test_records_carry_points_per_game_and_head_to_head(monkeypatch):
    monkeypatch.setattr(evaluate_versus, "measure_inference", lambda factory: {"encode_us": 0.0, "decide_us": 1.0, "total_us": 1.0})
    entrants = _entrants()

    records = evaluate_versus.evaluate_all(entrants, None, {"cpu": "test"}, games=4)

    assert {r.entrant_id for r in records} == {e["entrant_id"] for e in entrants}
    for record in records:
        quality, versus = record.metrics["quality"], record.metrics["versus"]
        assert record.protocol == "checkers.versus.v1" and record.game == "checkers"
        assert quality["n"] == 4 * (len(entrants) - 1)  # every other entrant, 4 games each
        assert 0.0 <= quality["mean"] <= 1.0
        assert versus["wins"] + versus["draws"] + versus["losses"] == quality["n"]
        assert quality["mean"] == round((versus["wins"] + 0.5 * versus["draws"]) / quality["n"], 4)
        assert set(versus["by_opponent"]) == {e["entrant_id"] for e in entrants if e["entrant_id"] != record.entrant_id}
    # 4 games per pair is far too few to order the weak baselines, but not to see the strong one.
    by_id = {r.entrant_id: r.metrics["quality"]["mean"] for r in records}
    assert by_id["baseline:material-2"] > max(by_id["baseline:random"], by_id["baseline:first-legal"])


def test_champion_entrants_reads_neat_and_layered_runs_with_their_search_depth_and_skips_smoke_tests(tmp_path):
    import time

    from evolve import InnovationTracker, initial_genome
    from telemetry import FileArtifactStore, FileMetricsStore, GenerationStats, SqliteRunRegistry

    registry = SqliteRunRegistry(tmp_path / "runs.db")
    metrics = FileMetricsStore(tmp_path / "metrics")
    artifacts = FileArtifactStore(tmp_path / "artifacts")

    def make_run(generations, champion_json, **config):
        run_id = registry.create_run(config={"game": "checkers", "interface": evaluate_versus.INTERFACE, **config})
        for g in range(generations):
            ref = f"{run_id}-gen{g}"
            artifacts.put_program(ref, champion_json.encode())
            metrics.record_generation(
                GenerationStats(run_id=run_id, island_id=None, generation=g, timestamp=time.time(), best_fitness=0.0, mean_fitness=0.0, worst_fitness=0.0, diversity=0.0, champion_ref=ref)
            )
        registry.update_status(run_id, "completed")
        return run_id

    neat_json = initial_genome(32, 1, InnovationTracker(first_hidden_id=34), __import__("random").Random(0)).to_json()
    layered_json = WeightVector(weights=tuple([0.1] * 33), layer_sizes=(32, 1)).to_json()
    neat = make_run(10, neat_json, representation="neat", search_depth=3, selection="speciation")
    layered = make_run(10, layered_json, representation="neuroevolution", selection="lexicase")
    make_run(3, layered_json, representation="neuroevolution")  # a smoke test: too short to be an entrant

    entrants = {e["entrant_id"]: e for e in evaluate_versus.champion_entrants(registry, metrics, artifacts)}

    assert set(entrants) == {f"run:{neat}", f"run:{layered}"}
    assert entrants[f"run:{neat}"]["search_depth"] == 3 and "NEAT" in entrants[f"run:{neat}"]["label"] and "3-ply" in entrants[f"run:{neat}"]["label"]
    assert entrants[f"run:{layered}"]["search_depth"] == 1
    # Both play: the factories build a working strategy through the game core.
    env = evaluate_versus.Checkers()
    env.reset()
    for entrant in entrants.values():
        assert entrant["factory"](env, evaluate_versus.random.Random(0))(None, env.legal_moves()) in env.legal_moves()
