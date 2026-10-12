from pathlib import Path

import pytest
from jobcore.specs import PublishSpec, load_spec

from publisher import publish
from publisher.__main__ import main

SPECS = sorted((Path(__file__).parents[1] / "specs").glob("*.yaml"))


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_example_spec_resolves(path):
    spec = load_spec(path)
    assert isinstance(spec, PublishSpec) and spec.target == path.stem
    spec.resolve()


def test_publishing_an_empty_field_writes_an_empty_catalog(tmp_path):
    summary = main([str(SPECS[0].parent / "snake.yaml")])
    assert summary["target"] == "snake" and summary["entries"] == [] and "garbage_collected" in summary
    assert (tmp_path / "models").exists()


def test_an_unknown_target_is_refused():
    with pytest.raises(ValueError, match="unknown publish target"):
        publish({"target": "checkers"})
