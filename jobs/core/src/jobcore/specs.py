"""Job payloads (docs/design/0018, decision 1): different work is a different payload to the same endpoint, not a new
script.

A `TrainSpec` says what one training run is -- game, interface, algorithm, the algorithm's own params, a budget, a
seed -- and nothing about how it's run. Each algorithm registers an `Algorithm`: a pydantic model for its params and
the budget units it understands. `TrainSpec.resolve()` checks a spec against that registration, so a bad spec fails
before a worker starts, with a message that names the field. `schemas()` is what `GET /specs/schemas`, CLI help and
MCP tool definitions are generated from.

Pure pydantic: the backend and the CLI import this to validate a spec without importing any trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

BudgetUnit = Literal["generations", "iterations", "games", "env_steps", "steps"]


class InitFrom(BaseModel):
    """Continue from an earlier run's final network: by run id (or a unique prefix), or by the run an experiment's
    arm produced for a seed (resolved by the scheduler, docs/design/0018 decision 2)."""

    model_config = ConfigDict(extra="forbid")

    run: str | None = None
    experiment: str | None = None
    arm: str | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _one_way(self) -> InitFrom:
        by_run, by_arm = self.run is not None, self.experiment is not None or self.arm is not None
        if by_run == by_arm:
            raise ValueError("init_from takes either `run`, or `experiment` and `arm` (and optionally `seed`)")
        if by_arm and (self.experiment is None or self.arm is None):
            raise ValueError("init_from by experiment needs both `experiment` and `arm`")
        return self


class TrainSpec(BaseModel):
    """One training run."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["train"] = "train"
    game: str = Field(..., min_length=1)
    interface: str | None = Field(None, description="The interface id (docs/design/0007); None = the game's default.")
    algorithm: str = Field(..., min_length=1, description="A registered algorithm, e.g. `neuroevolution`, `rl.dqn`.")
    params: dict[str, Any] = Field(default_factory=dict, description="The algorithm's own settings (its schema).")
    budget: dict[BudgetUnit, PositiveInt] = Field(..., description="Exactly one unit, e.g. {generations: 250}.")
    seed: int = 0
    init_from: InitFrom | None = None
    held_out_every: PositiveInt | None = Field(None, description="Record a held-out score every N; None = the default.")
    tags: dict[str, str] = Field(default_factory=dict, description="Recorded in the run config, e.g. experiment/arm.")

    @model_validator(mode="after")
    def _one_budget_unit(self) -> TrainSpec:
        if len(self.budget) != 1:
            raise ValueError(f"budget takes exactly one unit, got {sorted(self.budget) or 'none'}")
        return self

    @property
    def budget_unit(self) -> BudgetUnit:
        return next(iter(self.budget))

    @property
    def budget_amount(self) -> int:
        return next(iter(self.budget.values()))

    def resolve(self) -> tuple[Algorithm, BaseModel]:
        """The registered algorithm and this spec's params validated against it; raises ValueError if either is wrong."""
        algorithm = get_algorithm(self.algorithm, self.game)
        if self.budget_unit not in algorithm.budget_units:
            raise ValueError(
                f"{self.algorithm} is budgeted in {' or '.join(sorted(algorithm.budget_units))}, not {self.budget_unit}"
            )
        if algorithm.interfaces is not None and self.interface not in (None, *algorithm.interfaces):
            raise ValueError(
                f"{self.algorithm} on {self.game} trains under {' or '.join(sorted(algorithm.interfaces))}, "
                f"not {self.interface}"
            )
        return algorithm, algorithm.params.model_validate(self.params)


@dataclass(frozen=True)
class Algorithm:
    """One algorithm for some games. A name can be registered more than once for disjoint games -- neuroevolution on
    Snake and on Checkers share the idea and the name, not their settings."""

    name: str
    params: type[BaseModel]
    budget_units: frozenset[str]
    games: frozenset[str] | None = None  # None: any game
    summary: str = ""
    interfaces: frozenset[str] | None = None  # None: any of the game's interfaces

    def trains(self, game: str) -> bool:
        return self.games is None or game in self.games


_ALGORITHMS: dict[str, list[Algorithm]] = {}


def register_algorithm(algorithm: Algorithm) -> Algorithm:
    registered = _ALGORITHMS.setdefault(algorithm.name, [])
    if algorithm in registered:
        return algorithm  # the same registration again (a re-import) is fine
    for other in registered:
        if other.games is None or algorithm.games is None or other.games & algorithm.games:
            raise ValueError(f"algorithm {algorithm.name!r} is already registered differently for these games")
    registered.append(algorithm)
    return algorithm


def _load_builtin() -> None:
    import jobcore.algorithms
    import jobcore.protocols  # noqa: F401 -- and every protocol and publish target


def get_algorithm(name: str, game: str | None = None) -> Algorithm:
    """The registration of `name` for `game` (or its only registration, when `game` is None)."""
    _load_builtin()
    registered = _ALGORITHMS.get(name)
    if not registered:
        known = ", ".join(sorted(_ALGORITHMS)) or "none registered yet"
        raise ValueError(f"unknown algorithm {name!r} (known: {known})")
    if game is None:
        if len(registered) > 1:
            raise ValueError(f"algorithm {name!r} is registered for several games; say which")
        return registered[0]
    for algorithm in registered:
        if algorithm.trains(game):
            return algorithm
    games = sorted(g for a in registered for g in (a.games or ()))
    raise ValueError(f"{name} trains {', '.join(games)}, not {game}")


def algorithms() -> list[Algorithm]:
    _load_builtin()
    return [a for name in sorted(_ALGORITHMS) for a in _ALGORITHMS[name]]


class EvalSpec(BaseModel):
    """One evaluation: a leaderboard protocol (records every entrant into the evaluation store) or a report (returns
    a result, e.g. a head-to-head SPRT). The protocol is versioned and fixed; `params` are only what it lets vary."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["evaluate"] = "evaluate"
    protocol: str = Field(..., min_length=1, description="A registered protocol, e.g. `checkers.versus.v2`.")
    params: dict[str, Any] = Field(default_factory=dict)

    def resolve(self) -> tuple[Protocol, BaseModel]:
        protocol = _get(_PROTOCOLS, "protocol", self.protocol)
        return protocol, protocol.params.model_validate(self.params)


class PublishSpec(BaseModel):
    """Export to the model store: a target is what gets packaged (a game's leaderboard champions, TinyLM checkpoints,
    the scale-test model)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["publish"] = "publish"
    target: str = Field(..., min_length=1, description="A registered publish target, e.g. `snake`.")
    params: dict[str, Any] = Field(default_factory=dict)

    def resolve(self) -> tuple[PublishTarget, BaseModel]:
        target = _get(_TARGETS, "publish target", self.target)
        return target, target.params.model_validate(self.params)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` on top of `base`, recursing into nested dicts (so an arm's `params.agent.hidden` changes one setting,
    not the whole `agent`)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ArmSpec(BaseModel):
    """One arm: TrainSpec fields on top of the experiment's `defaults`, and what it is compared against."""

    model_config = ConfigDict(extra="forbid")

    train: dict[str, Any] = Field(default_factory=dict, description="TrainSpec fields overriding the defaults.")
    baseline: str | None = Field(None, description="The arm it differs from in one thing: `arm` or `experiment/arm`.")
    init_from: str | None = Field(None, description="Continue `experiment/arm`'s run of the same seed.")
    seeds: list[int] | None = Field(None, min_length=1, description="This arm's seeds, if not the experiment's.")


class ExperimentSpec(BaseModel):
    """A tracked comparison (docs/design/0018, decision 2): arms x seeds of training runs, each tagged with the
    experiment and arm, then optionally a report. The scheduler fans it out into trainer jobs and resumes it: an arm x
    seed already completed is never trained again."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["experiment"] = "experiment"
    name: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    seeds: list[int] = Field(..., min_length=1)
    defaults: dict[str, Any] = Field(default_factory=dict, description="TrainSpec fields every arm shares.")
    arms: dict[str, ArmSpec] = Field(..., min_length=1)
    report: dict[str, Any] | None = Field(
        None, description="An EvalSpec to run once every arm x seed is done (`params.experiment` defaults to this one)."
    )

    def expand(self) -> list[tuple[str, int, TrainSpec]]:
        """Every (arm, seed, TrainSpec) the experiment trains, in seed-major order (a partial experiment is then a
        complete one with fewer seeds)."""
        out = []
        for seed in sorted({seed for arm in self.arms.values() for seed in arm.seeds or self.seeds}):
            for name, arm in self.arms.items():
                if seed not in (arm.seeds or self.seeds):
                    continue
                data = deep_merge(self.defaults, arm.train)
                data = deep_merge(data, {"seed": seed, "tags": {"experiment": self.name, "arm": name}})
                if arm.init_from:
                    experiment, _, parent = arm.init_from.rpartition("/")
                    data["init_from"] = {"experiment": experiment or self.name, "arm": parent, "seed": seed}
                out.append((name, seed, TrainSpec.model_validate(data)))
        return out

    def resolve(self) -> tuple[ExperimentSpec, list[tuple[str, int, TrainSpec]]]:
        """Checks every arm x seed resolves, every baseline names an arm, and the report resolves."""
        expanded = self.expand()
        for _arm, _seed, spec in expanded:
            spec.resolve()
        for name, arm in self.arms.items():
            if arm.baseline and "/" not in arm.baseline and arm.baseline not in self.arms:
                raise ValueError(f"arm {name}'s baseline {arm.baseline!r} isn't an arm of {self.name}")
        if self.report is not None:
            self.report_spec().resolve()
        return self, expanded

    def report_spec(self) -> EvalSpec | None:
        """The report as an EvalSpec; a protocol that takes an `experiment` parameter gets this one by default."""
        if self.report is None:
            return None
        spec = EvalSpec.model_validate(self.report)
        protocol = _PROTOCOLS.get(spec.protocol)
        if protocol is not None and "experiment" in protocol.params.model_fields and "experiment" not in spec.params:
            spec = spec.model_copy(update={"params": {"experiment": self.name, **spec.params}})
        return spec


@dataclass(frozen=True)
class Protocol:
    name: str
    game: str
    params: type[BaseModel]
    output: Literal["leaderboard", "report"]
    summary: str = ""


@dataclass(frozen=True)
class PublishTarget:
    name: str
    params: type[BaseModel]
    summary: str = ""


_PROTOCOLS: dict[str, Protocol] = {}
_TARGETS: dict[str, PublishTarget] = {}


def _register(registry: dict[str, Any], what: str, item: Any) -> Any:
    existing = registry.get(item.name)
    if existing is not None and existing != item:
        raise ValueError(f"{what} {item.name!r} is already registered differently")
    registry[item.name] = item
    return item


def _get(registry: dict[str, Any], what: str, name: str) -> Any:
    _load_builtin()
    try:
        return registry[name]
    except KeyError:
        raise ValueError(f"unknown {what} {name!r} (known: {', '.join(sorted(registry)) or 'none'})") from None


def register_protocol(protocol: Protocol) -> Protocol:
    return _register(_PROTOCOLS, "protocol", protocol)


def register_publish_target(target: PublishTarget) -> PublishTarget:
    return _register(_TARGETS, "publish target", target)


Spec = TrainSpec | EvalSpec | PublishSpec | ExperimentSpec
_KINDS: dict[str, type[BaseModel]] = {
    "train": TrainSpec,
    "evaluate": EvalSpec,
    "publish": PublishSpec,
    "experiment": ExperimentSpec,
}


def parse_spec(data: dict[str, Any]) -> Spec:
    """Any job payload, by its `kind` (default `train`)."""
    kind = data.get("kind", "train")
    if kind not in _KINDS:
        raise ValueError(f"unknown spec kind {kind!r} (known: {', '.join(_KINDS)})")
    return _KINDS[kind].model_validate(data)  # type: ignore[return-value]


def load_spec(source: str | Path) -> Spec:
    """A spec of any kind from a .yaml/.yml or .json file."""
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    return parse_spec(yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text))


def schemas() -> dict[str, Any]:
    """JSON Schema for every payload kind, and every registered algorithm's, protocol's and publish target's params
    (algorithms keyed `name`, or `name@game` when a name is registered per game)."""

    def key(a: Algorithm) -> str:
        return a.name if len(_ALGORITHMS[a.name]) == 1 else f"{a.name}@{','.join(sorted(a.games or ()))}"

    _load_builtin()
    return {
        "kinds": {kind: model.model_json_schema() for kind, model in _KINDS.items()},
        "algorithms": {
            key(a): {
                "name": a.name,
                "summary": a.summary,
                "budget_units": sorted(a.budget_units),
                "games": sorted(a.games) if a.games is not None else None,
                "params": a.params.model_json_schema(),
            }
            for a in algorithms()
        },
        "protocols": {
            p.name: {"game": p.game, "output": p.output, "summary": p.summary, "params": p.params.model_json_schema()}
            for p in sorted(_PROTOCOLS.values(), key=lambda p: p.name)
        },
        "publish_targets": {
            t.name: {"summary": t.summary, "params": t.params.model_json_schema()}
            for t in sorted(_TARGETS.values(), key=lambda t: t.name)
        },
    }
