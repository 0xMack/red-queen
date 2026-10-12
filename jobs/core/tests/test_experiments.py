import pytest
from pydantic import ValidationError

from jobcore.specs import ExperimentSpec, deep_merge, parse_spec

SPEC = {
    "kind": "experiment",
    "name": "probe-v1",
    "seeds": [0, 1],
    "defaults": {
        "game": "checkers",
        "algorithm": "td_lambda",
        "budget": {"iterations": 4},
        "params": {"games_per_iteration": 10, "agent": {"hidden": 8, "pool_every": 50}},
    },
    "arms": {
        "base": {},
        "wide": {"train": {"params": {"agent": {"hidden": 16}}}, "baseline": "base"},
        "ft": {"train": {"params": {"agent": {"search_depth": 2}}}, "baseline": "base", "init_from": "probe-v0/base"},
    },
    "report": {"protocol": "checkers.static-round-robin", "params": {"games": 2}},
}


def test_an_experiment_expands_to_tagged_seeded_train_specs():
    experiment = parse_spec(SPEC)
    assert isinstance(experiment, ExperimentSpec)
    _, expanded = experiment.resolve()
    assert [(arm, seed) for arm, seed, _ in expanded] == [
        ("base", 0),
        ("wide", 0),
        ("ft", 0),
        ("base", 1),
        ("wide", 1),
        ("ft", 1),
    ]
    wide = next(spec for arm, seed, spec in expanded if arm == "wide" and seed == 1)
    assert wide.seed == 1 and wide.tags == {"experiment": "probe-v1", "arm": "wide"}
    assert wide.params["agent"] == {"hidden": 16, "pool_every": 50}  # one setting changed, the rest kept
    ft = next(spec for arm, seed, spec in expanded if arm == "ft" and seed == 0)
    assert ft.init_from.experiment == "probe-v0" and ft.init_from.arm == "base" and ft.init_from.seed == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"arms": {"x": {"baseline": "nope"}}}, "isn't an arm"),
        ({"arms": {"x": {"train": {"algorithm": "nope"}}}}, "unknown algorithm"),
        ({"report": {"protocol": "nope"}}, "unknown protocol"),
    ],
)
def test_an_experiment_that_cant_run_is_refused_whole(change, message):
    with pytest.raises(ValueError, match=message):
        ExperimentSpec.model_validate({**SPEC, **change}).resolve()


def test_names_and_seeds_are_checked():
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**SPEC, "name": "has spaces"})
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**SPEC, "seeds": []})


def test_deep_merge_recurses_into_dicts_only():
    assert deep_merge({"a": {"b": 1, "c": 2}, "l": [1]}, {"a": {"b": 3}, "l": [2]}) == {"a": {"b": 3, "c": 2}, "l": [2]}
