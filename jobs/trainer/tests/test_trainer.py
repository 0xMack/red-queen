from pathlib import Path

import pytest
from jobcore.specs import load_spec
from telemetry import SqliteRunRegistry

from trainer import train
from trainer.__main__ import main
from trainer.registry import _ADAPTERS

SPECS = sorted((Path(__file__).parents[1] / "specs").glob("*.yaml"))


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_example_spec_resolves_to_an_adapter(path):
    import trainer.algorithms  # noqa: F401

    spec = load_spec(path)
    algorithm, params = spec.resolve()
    assert (spec.algorithm, spec.game) in _ADAPTERS
    assert params == algorithm.params()  # the examples are the defaults: they set nothing the model doesn't default


def test_every_registered_algorithm_has_an_example_spec_and_an_adapter():
    from jobcore.specs import algorithms

    import trainer.algorithms  # noqa: F401

    examples = {(s.algorithm, s.game) for s in map(load_spec, SPECS)}
    for algorithm in algorithms():
        for game in algorithm.games or ():
            assert (algorithm.name, game) in _ADAPTERS
            assert (algorithm.name, game) in examples


def test_the_command_line_runs_a_spec_with_overrides(tmp_path, capsys):
    run_id = main([str(SPECS[0].parent / "gp.yaml"), "--set", "budget.generations=2", "--set", "tags.experiment=probe"])
    run = SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id)
    assert run.status == "completed" and run.config["generations"] == 2 and run.config["experiment"] == "probe"
    assert capsys.readouterr().out.strip().endswith(run_id)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"game": "go", "algorithm": "neat"}, "trains checkers, snake, not go"),
        ({"game": "checkers", "algorithm": "neat", "interface": "checkers/other.v1"}, "trains under checkers/board32"),
        ({"game": "snake", "algorithm": "neat", "params": {"neat": {"speciatoin": False}}}, "unknown NeatConfig"),
    ],
)
def test_specs_it_cant_run_are_refused_before_a_run_is_recorded(tmp_path, spec, message):
    with pytest.raises(ValueError, match=message):
        train({"budget": {"generations": 1}, **spec})
    assert SqliteRunRegistry(tmp_path / "runs.db").list_runs() == []
