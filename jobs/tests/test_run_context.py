import pytest
import run_context
from costs import TrainingCostMeter
from run_context import TelemetryStores, recorded_run
from telemetry import GenerationStats


def _stats(run_id: str, generation: int) -> GenerationStats:
    return GenerationStats(
        run_id=run_id,
        island_id=None,
        generation=generation,
        timestamp=1.0,
        best_fitness=float(generation),
        mean_fitness=0.0,
        worst_fitness=0.0,
        diversity=0.0,
        champion_ref=f"{run_id}-gen{generation}",
        held_out_score=1.5,
    )


def test_a_run_that_finishes_is_completed_with_its_config_and_summary(tmp_path):
    stores = TelemetryStores.open(tmp_path)
    with recorded_run({"game": "snake"}, stores) as run:
        for generation in range(2):
            run.metrics.record_generation(_stats(run.run_id, generation))
        history = run.set_training_summary(TrainingCostMeter(population_size=4))

    info = stores.registry.get_run(run.run_id)
    assert info.status == "completed" and info.config == {"game": "snake"}
    assert [h.generation for h in history] == [0, 1]
    assert info.summary["best_fitness"] == 1.0 and info.summary["held_out_score"] == 1.5
    assert info.summary["cost"]["measured"] is True


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_a_run_that_dies_is_marked_failed_and_the_error_propagates(tmp_path, error):
    stores = TelemetryStores.open(tmp_path)
    with pytest.raises(type(error)), recorded_run({}, stores) as run:
        raise error

    assert stores.registry.get_run(run.run_id).status == "failed"


def test_stores_default_to_the_patchable_run_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    with recorded_run({}) as run:
        pass
    assert TelemetryStores.open(tmp_path).registry.get_run(run.run_id).status == "completed"


def test_the_control_callback_returns_immediately_while_running(tmp_path):
    with recorded_run({}, TelemetryStores.open(tmp_path)) as run:
        cost = TrainingCostMeter(population_size=1)
        run.control_callback(cost)(None)
        run.control_callback()(None)
