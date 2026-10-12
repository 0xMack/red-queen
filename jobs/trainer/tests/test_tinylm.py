import io
import json

import numpy as np
import tinylm
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry

from trainer import train


def test_a_tinylm_run_is_recorded_beside_its_checkpoint(tmp_path):
    run_id = train(
        {
            "game": "text",
            "algorithm": "tinylm",
            "budget": {"steps": 5},
            "held_out_every": 2,
            "params": {"checkpoint": "t", "log_every": 2, "d_model": 16, "n_heads": 2, "n_layers": 1, "d_hidden": 32},
        }
    )

    run = SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id)
    assert run.status == "completed" and run.config["paradigm"] == "supervised" and run.config["game"] == "text"
    history = FileMetricsStore(tmp_path / "metrics").history(run_id)
    # 5 steps in blocks of 2: generations end at steps 2, 4, 5; held out every 2nd generation and on the last
    assert [h.extras["step"] for h in history] == [2.0, 4.0, 5.0]
    assert [h.held_out_score is not None for h in history] == [True, False, True]
    assert all(h.best_fitness >= h.mean_fitness >= h.worst_fitness and h.mean_fitness < 0 for h in history)

    # the checkpoint the model store publishes, and the run's own copy of the weights, are the same model
    model, _, meta = tinylm.load(tmp_path / "tinylm" / "t")
    assert meta["held_out_loss"] == round(-history[-1].held_out_score, 4) == -run.summary["held_out_score"]
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    with np.load(io.BytesIO(artifacts.get_program(f"{run_id}-weights"))) as weights:
        assert set(weights.files) == set(tinylm.checkpoint.named_parameters(model))
    assert json.loads(artifacts.get_program(history[-1].champion_ref))["step"] == 5
