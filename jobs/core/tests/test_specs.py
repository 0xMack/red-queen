import pytest
from pydantic import BaseModel, ValidationError

from jobcore.specs import Algorithm, InitFrom, TrainSpec, get_algorithm, register_algorithm, schemas


class ToyParams(BaseModel):
    hidden: int = 16
    sigma: float = 0.2


TOY = register_algorithm(
    Algorithm("toy", ToyParams, budget_units=frozenset({"generations"}), games=frozenset({"snake"}), summary="a toy")
)


def spec(**overrides) -> TrainSpec:
    return TrainSpec(**{"game": "snake", "algorithm": "toy", "budget": {"generations": 5}, **overrides})


def test_a_spec_resolves_to_its_algorithm_and_validated_params():
    algorithm, params = spec(params={"hidden": 8}).resolve()
    assert algorithm is TOY and params == ToyParams(hidden=8, sigma=0.2)
    assert (spec().budget_unit, spec().budget_amount) == ("generations", 5)


def test_a_spec_parses_from_the_yaml_shape():
    parsed = TrainSpec.model_validate(
        {"game": "snake", "algorithm": "toy", "budget": {"generations": 3}, "init_from": {"run": "672890ba"}}
    )
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
        ({"game": "checkers"}, "trains snake, not checkers"),
    ],
)
def test_specs_the_algorithm_cant_run_are_refused_before_any_worker_starts(overrides, message):
    with pytest.raises(ValueError, match=message):
        spec(**overrides).resolve()


def test_bad_params_are_refused_by_the_algorithms_own_schema():
    with pytest.raises(ValidationError, match="hidden"):
        spec(params={"hidden": "wide"}).resolve()


def test_registering_a_different_algorithm_under_a_taken_name_is_refused():
    assert register_algorithm(TOY) is TOY  # the same registration again is fine (a re-import)
    with pytest.raises(ValueError, match="already registered"):
        register_algorithm(Algorithm("toy", ToyParams, budget_units=frozenset({"games"})))
    assert get_algorithm("toy") is TOY


def test_schemas_describe_every_kind_and_algorithm():
    described = schemas()
    assert described["kinds"]["train"]["title"] == "TrainSpec"
    toy = described["algorithms"]["toy"]
    assert toy["budget_units"] == ["generations"] and set(toy["params"]["properties"]) == {"hidden", "sigma"}
