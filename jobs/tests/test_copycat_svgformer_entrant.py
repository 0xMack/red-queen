"""A trained svgformer on the Copy Cat leaderboard (docs/design/0024 P3): a completed run outside an experiment becomes two
entrants, the model with polish and the model-or-tracer pick, and an experiment's runs stay off."""

import pytest

pytest.importorskip("torch")

from arena.copycat import load_set
from evaluator.protocols.copycat_clean import record_for, svgformer_entrants
from games.copycat import Svg
from jobcore import open_sink
from trainer import train

XS = {
    "game": "copycat",
    "algorithm": "copycat.svgformer",
    "budget": {"steps": 2},
    "params": {
        "model": "XS",
        "batch": 2,
        "workers": 0,
        "log_every": 1,
        "warmup": 1,
        "validate_images": 2,
        "device": "cpu",
    },
}


def test_a_run_is_two_entrants_and_an_experiment_s_none():
    run_id = train(XS)
    train({**XS, "tags": {"experiment": "e", "arm": "a"}})
    sink = open_sink()
    entrants = svgformer_entrants(sink)
    assert [e["entrant_id"] for e in entrants] == [f"run:{run_id}", f"run:{run_id}+structured"]

    targets = load_set()[:2]
    alone, paired = (record_for("copycat.clean.v1", e, targets, {}) for e in entrants)
    assert (alone.entrant_kind, alone.run_id, alone.champion_ref) == ("champion", run_id, f"{run_id}-weights")
    for record in (alone, paired):
        images = record.metrics["copycat"]["images"]
        assert len(images) == 2 and all(Svg(i["svg"]).width == 128 for i in images)
        assert record.metrics["inference"]["model_bytes"] > 0
        assert record.metrics["training"]["measured"]
    picked = paired.metrics["copycat"].get("picked_tracer", 0)
    assert 0 <= picked <= 2 and paired.metrics["copycat"]["settings"]["paired_with"] == "structured"
