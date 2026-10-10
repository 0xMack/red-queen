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
    import jobcore.algorithms  # noqa: F401 -- registers every built-in algorithm's params


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


def load_spec(source: str | Path) -> TrainSpec:
    """A spec from a .yaml/.yml or .json file."""
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    return TrainSpec.model_validate(data)


def schemas() -> dict[str, Any]:
    """JSON Schema for every payload kind and every registered algorithm's params (keyed `name` or `name@game`)."""

    def key(a: Algorithm) -> str:
        return a.name if len(_ALGORITHMS[a.name]) == 1 else f"{a.name}@{','.join(sorted(a.games or ()))}"

    return {
        "kinds": {"train": TrainSpec.model_json_schema()},
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
    }
