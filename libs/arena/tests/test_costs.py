import time

from arena.costs import TrainingCostMeter, hardware_fingerprint


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
