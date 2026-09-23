import time

import evaluate
from costs import TrainingCostMeter, hardware_fingerprint
from games import interfaces
from telemetry import FileMetricsStore, GenerationStats, RunInfo

FEATURES = interfaces.get(evaluate.FEATURES)


def test_score_stats_reports_spread_not_just_the_mean():
    stats = evaluate.score_stats([0, 2, 4, 10])

    assert stats["mean"] == 4.0
    assert stats["median"] == 3.0
    assert (stats["min"], stats["max"]) == (0, 10)
    assert stats["zero_rate"] == 0.25
    assert stats["ci95"] > 0


def test_greedy_baseline_beats_random_on_held_out_seeds(monkeypatch):
    monkeypatch.setattr(evaluate, "HELD_OUT_SEEDS", tuple(range(10_000, 10_020)))

    greedy = evaluate.measure_quality(FEATURES, evaluate.greedy_policy_factory, training_seeds=[0])
    random_ = evaluate.measure_quality(FEATURES, evaluate.random_policy_factory, training_seeds=[0])

    assert greedy["mean"] > random_["mean"] + 5
    assert greedy["n"] == 20
    assert abs(greedy["generalization_gap"] - (greedy["train_mean"] - greedy["mean"])) < 1e-3


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


def test_cost_meter_excludes_time_spent_paused():
    meter = TrainingCostMeter(population_size=4)
    pausing_control = meter.excluding_pauses(lambda _summary: time.sleep(0.05))

    for _ in range(2):
        meter.on_generation(None)
        pausing_control(None)
    summary = meter.summary()

    assert summary["fitness_evaluations"] == 8
    assert summary["paused_s"] >= 0.1
    assert summary["active_s"] < summary["wall_s"] - 0.09
    assert summary["hardware"] == hardware_fingerprint()
