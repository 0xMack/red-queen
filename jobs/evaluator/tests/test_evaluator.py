from pathlib import Path

import pytest
from jobcore.specs import EvalSpec, load_spec

from evaluator import evaluate
from evaluator.__main__ import main
from evaluator.registry import _RUNNERS

SPECS = sorted((Path(__file__).parents[1] / "specs").glob("*.yaml"))


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_example_spec_resolves_to_a_runner(path):
    import evaluator.protocols  # noqa: F401

    spec = load_spec(path)
    assert isinstance(spec, EvalSpec)
    spec.resolve()
    assert spec.protocol in _RUNNERS and path.stem == spec.protocol


def test_every_registered_protocol_has_a_runner_and_an_example():
    from jobcore.specs import schemas

    import evaluator.protocols  # noqa: F401

    assert set(schemas()["protocols"]) == set(_RUNNERS) == {p.stem for p in SPECS}


def test_a_report_comes_back_from_the_command_line(capsys):
    spec = SPECS[0].parent / "checkers.static-round-robin.yaml"
    result = main([str(spec), "--set", "params.games=2"])
    assert len(result) == 15 and all(sum(t.values()) == 2 for t in result.values())  # 6 strategies, 2 games each
    assert '"random vs first-legal"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"protocol": "snake.score.v9"}, "unknown protocol"),
        ({"protocol": "checkers.sprt", "params": {"a": "material-2"}}, "b"),
        ({"protocol": "snake.score.v2", "params": {"seeds": 3}}, "Extra inputs"),
    ],
)
def test_specs_it_cant_run_are_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        evaluate(spec)
