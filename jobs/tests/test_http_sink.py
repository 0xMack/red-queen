"""A job on the HTTP sink records exactly what it records on the local one (docs/design/0018, decision 3).

Golden cases run through `jobcore.http` against the backend itself (its TestClient), the backend writing the same data
directory the digest then reads: the same configs, curves, champions and evaluation records as the fixtures, so every
write path a job uses -- runs, status, summaries, generations, artifacts, evaluations and their pruning -- goes over
HTTP and lands identically.
"""

import pytest
from golden_cases import CASES, load_fixture, run_case
from golden_eval import EVAL_CASES, EvalCase, matches
from golden_eval import load_fixture as load_eval_fixture
from golden_eval import run_case as run_eval_case

TRAINING = ["gp", "snake-neuro-lexicase", "rl-q_learning", "selfplay-td", "pbt", "tinylm"]
EVALUATIONS = ["bandit.skill.v1", "checkers.versus.v2", "checkers.sprt"]


def _http_sink():
    """A fresh backend over the current data directory (its stores are cached per process), and a sink over it."""
    from backend.dependencies import _stores
    from backend.main import app
    from fastapi.testclient import TestClient
    from jobcore.http import http_sink

    _stores.cache_clear()
    return http_sink(TestClient(app))


@pytest.mark.parametrize("name", TRAINING)
def test_a_training_run_over_http_is_its_golden_run(name):
    from trainer import train

    expected = load_fixture()[name]
    assert run_case(name, runner=lambda case: train(case.spec, sink=_http_sink())) == expected


@pytest.mark.parametrize("name", EVALUATIONS)
def test_an_evaluation_over_http_is_its_golden_evaluation(name):
    from evaluator import evaluate

    case = EVAL_CASES[name]
    protocol, params = _evaluation(name)
    over_http = EvalCase(lambda: evaluate({"protocol": protocol, "params": params}, sink=_http_sink()), case.digest)
    assert matches(name, run_eval_case(name, over_http), load_eval_fixture()[name])


def _evaluation(name: str) -> tuple[str, dict]:
    """The protocol and params each golden case evaluates (golden_eval.EVAL_CASES)."""
    return name, ({"a": "material-2", "b": "random"} if name == "checkers.sprt" else {})


def test_the_cases_exist():
    assert set(TRAINING) <= set(CASES) and set(EVALUATIONS) <= set(EVAL_CASES)
