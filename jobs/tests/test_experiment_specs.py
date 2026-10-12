"""Every checked-in experiment (experiments/*.yaml) resolves: its arms are valid trainer specs, its baselines name
arms, and its report is a valid evaluator spec -- so a renamed parameter can't silently strand one."""

from pathlib import Path

import evaluator.protocols  # noqa: F401  (registers the report protocols)
import pytest
import trainer.algorithms  # noqa: F401  (registers the algorithms)
from jobcore.specs import ExperimentSpec, PipelineSpec, ScheduleSpec, load_spec

EXPERIMENTS = sorted((Path(__file__).resolve().parents[2] / "experiments").glob("*.yaml"))


def test_there_are_experiments():
    assert len(EXPERIMENTS) >= 19


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_an_experiment_resolves(path):
    spec = load_spec(path)
    assert isinstance(spec, ExperimentSpec) and spec.name == path.stem
    spec.resolve()


PIPELINES = sorted((Path(__file__).resolve().parents[2] / "experiments" / "pipelines").glob("*.yaml"))


@pytest.mark.parametrize("path", PIPELINES, ids=lambda p: p.stem)
def test_a_pipeline_resolves(path):
    spec = load_spec(path)
    assert isinstance(spec, PipelineSpec) and spec.name == path.stem
    spec.resolve()


SCHEDULES = sorted((Path(__file__).resolve().parents[2] / "experiments" / "schedules").glob("*.yaml"))


@pytest.mark.parametrize("path", SCHEDULES, ids=lambda p: p.stem)
def test_a_schedule_resolves(path):
    spec = load_spec(path)
    assert isinstance(spec, ScheduleSpec) and spec.name == path.stem
    spec.resolve()
