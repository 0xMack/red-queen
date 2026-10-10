import evaluate
from games import interfaces
from telemetry import FileMetricsStore, GenerationStats, RunInfo

FEATURES = interfaces.get(evaluate.FEATURES)


def test_inference_is_split_into_encode_and_decide():
    inference = evaluate.measure_inference(FEATURES, evaluate.greedy_policy_factory, repeats=2, decisions=50)

    assert inference["encode_us"] > 0 and inference["decide_us"] > 0
    assert abs(inference["total_us"] - inference["encode_us"] - inference["decide_us"]) < 0.01


def test_legacy_run_training_cost_is_estimated_and_labelled(tmp_path):
    metrics = FileMetricsStore(tmp_path / "metrics")
    for generation in range(3):
        metrics.record_generation(
            GenerationStats(
                run_id="r1",
                island_id=None,
                generation=generation,
                timestamp=100.0 + generation * 10,
                best_fitness=0.0,
                mean_fitness=0.0,
                worst_fitness=0.0,
                diversity=0.0,
                champion_ref=f"r1-gen{generation}",
            )
        )
    run = RunInfo(run_id="r1", config={"population_size": 10}, status="completed", created_at=1.0, updated_at=2.0)

    cost = evaluate.training_cost(run, metrics)

    assert cost["measured"] is False
    assert cost["fitness_evaluations"] == 30  # 10 genomes x 3 generations
    assert cost["episodes"] == 150  # x 5 default training seeds
    assert cost["active_s"] == 20.0
