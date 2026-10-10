import pytest
from pydantic import BaseModel, ValidationError

from jobcore import specs
from jobcore.specs import Algorithm, InitFrom, TrainSpec, get_algorithm, load_spec, register_algorithm, schemas


class ToyParams(BaseModel):
    hidden: int = 16
    sigma: float = 0.2


class OtherParams(BaseModel):
    depth: int = 1


TOY = Algorithm("toy", ToyParams, budget_units=frozenset({"generations"}), games=frozenset({"snake"}), summary="a toy")


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """A registry of just the built-ins plus `toy`, so nothing registered here leaks into other packages' tests."""
    specs._load_builtin()
    monkeypatch.setattr(specs, "_ALGORITHMS", {name: list(algs) for name, algs in specs._ALGORITHMS.items()})
    register_algorithm(TOY)


def spec(**overrides) -> TrainSpec:
    return TrainSpec(**{"game": "snake", "algorithm": "toy", "budget": {"generations": 5}, **overrides})


def test_a_spec_resolves_to_its_algorithm_and_validated_params():
    algorithm, params = spec(params={"hidden": 8}).resolve()
    assert algorithm is TOY and params == ToyParams(hidden=8, sigma=0.2)
    assert (spec().budget_unit, spec().budget_amount) == ("generations", 5)


def test_a_spec_parses_from_the_yaml_shape(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("game: snake\nalgorithm: toy\nbudget: {generations: 3}\ninit_from: {run: 672890ba}\n")
    parsed = load_spec(path)
    assert parsed.init_from == InitFrom(run="672890ba") and parsed.kind == "train"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"budget": {}}, "exactly one unit"),
        ({"budget": {"generations": 5, "games": 5}}, "exactly one unit"),
        ({"budget": {"generations": 0}}, "greater than 0"),
        ({"budgte": {"generations": 5}}, "Extra inputs"),
        ({"init_from": {"run": "x", "arm": "y"}}, "either `run`"),
        ({"init_from": {"arm": "y"}}, "both `experiment` and `arm`"),
    ],
)
def test_malformed_specs_fail_with_a_message_naming_the_problem(overrides, message):
    with pytest.raises(ValidationError, match=message):
        spec(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"algorithm": "nope"}, "unknown algorithm 'nope'"),
        ({"budget": {"games": 5}}, "budgeted in generations, not games"),
        ({"game": "checkers"}, "toy trains snake, not checkers"),
    ],
)
def test_specs_the_algorithm_cant_run_are_refused_before_any_worker_starts(overrides, message):
    with pytest.raises(ValueError, match=message):
        spec(**overrides).resolve()


def test_bad_params_are_refused_by_the_algorithms_own_schema():
    with pytest.raises(ValidationError, match="hidden"):
        spec(params={"hidden": "wide"}).resolve()


def test_one_name_can_be_registered_per_game_with_its_own_params():
    other = register_algorithm(Algorithm("toy", OtherParams, frozenset({"games"}), games=frozenset({"checkers"})))
    assert get_algorithm("toy", "snake") is TOY and get_algorithm("toy", "checkers") is other
    _, params = spec(game="checkers", budget={"games": 3}, params={"depth": 2}).resolve()
    assert params == OtherParams(depth=2)
    with pytest.raises(ValueError, match="several games"):
        get_algorithm("toy")


def test_overlapping_registrations_are_refused():
    assert register_algorithm(TOY) is TOY  # the same registration again is fine (a re-import)
    with pytest.raises(ValueError, match="already registered differently"):
        register_algorithm(Algorithm("toy", OtherParams, frozenset({"generations"}), games=frozenset({"snake"})))
    with pytest.raises(ValueError, match="already registered differently"):
        register_algorithm(Algorithm("toy", OtherParams, frozenset({"generations"})))  # every game: overlaps snake


def test_schemas_describe_every_kind_and_algorithm():
    described = schemas()
    assert described["kinds"]["train"]["title"] == "TrainSpec"
    toy = described["algorithms"]["toy"]
    assert toy["budget_units"] == ["generations"] and set(toy["params"]["properties"]) == {"hidden", "sigma"}
    # a name registered per game is keyed by game too
    assert {"neuroevolution@checkers", "neuroevolution@snake", "gp"} <= set(described["algorithms"])
