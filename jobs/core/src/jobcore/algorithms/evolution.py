"""The evolution family: linear GP, neuroevolution and NEAT (Snake and Checkers), search distillation, and evolving a
bandit strategy's settings. Each is budgeted in generations."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from evolve import NeatConfig
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from jobcore.specs import Algorithm, register_algorithm

GENERATIONS = frozenset({"generations"})
NEAT_FIELDS = frozenset(f.name for f in dataclasses.fields(NeatConfig))


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NeatOverrides(_Params):
    neat: dict[str, Any] = Field(
        default_factory=dict, description="NeatConfig fields to override (on top of this game's defaults)."
    )

    @field_validator("neat")
    @classmethod
    def _known_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = set(value) - NEAT_FIELDS
        if unknown:
            raise ValueError(f"unknown NeatConfig fields: {sorted(unknown)}")
        return value


class GpParams(_Params):
    population_size: PositiveInt = 60
    num_instructions: PositiveInt = 12
    num_registers: PositiveInt = 4


class SnakeNeuroParams(_Params):
    hidden: PositiveInt = 16
    population_size: PositiveInt = 100
    max_steps: PositiveInt = Field(200, description="Training episode cap (the leaderboard plays 1000-step games).")
    seeds: str = Field("fixed:5", description="Training-seed strategy: fixed:N or resample:N (arena.seeding).")
    selection: Literal["lexicase", "tournament"] = "lexicase"
    tournament_k: PositiveInt = 4


class SnakeNeatParams(_NeatOverrides):
    population_size: PositiveInt = 100
    max_steps: PositiveInt = 200
    seeds: str = "fixed:5"


class _CheckersOpponents(_Params):
    opponents: list[str] = Field(["random", "material-1", "material-2"], min_length=1)
    depth: PositiveInt = Field(1, description="Alpha-beta plies the evaluator searches (1 = one ply).")
    monitor_games: PositiveInt = Field(10, description="Held-out games per opponent, seats alternating.")


class _CheckersMatchFitness(_CheckersOpponents):
    hall: int = Field(0, ge=0, description="Hall-of-fame size: past champions as extra opponents.")
    seed_material: float = Field(0.0, ge=0, le=1, description="Fraction of the population started as material.")
    margin: bool = Field(False, description="Score draws by the material edge held.")
    resample: bool = Field(False, description="Fresh opponent games every generation (nothing to memorize).")
    games_per_opponent: PositiveInt = 1


class CheckersNeuroParams(_CheckersMatchFitness):
    population_size: PositiveInt = 40
    hidden: PositiveInt = 8
    sigma: float = Field(0.2, gt=0)
    mutation_rate: float = Field(1.0, gt=0, le=1)
    opening_plies: int = Field(0, ge=0, description="Random moves opening every fitness game.")
    workers: PositiveInt = Field(1, description="Processes scoring a generation (the same result as 1, faster).")


class CheckersNeatParams(_CheckersMatchFitness, _NeatOverrides):
    population_size: PositiveInt = 60


class DistillParams(_CheckersOpponents):
    opponents: list[str] = Field(["material-3", "material-4"], min_length=1)
    depth: PositiveInt = 3
    population_size: PositiveInt = 100
    hidden: PositiveInt = 16
    label_depth: PositiveInt = Field(6, description="Plies of material search that label positions.")
    label_kind: Literal["search", "rollout", "blend"] = "search"
    pool_games: PositiveInt = 2000
    sample: PositiveInt = 600
    sigma: float = Field(0.1, gt=0)
    mutation_rate: float = Field(0.1, gt=0, le=1)
    workers: PositiveInt = 1


class BanditEvolveParams(_Params):
    scenarios: list[str] = Field(["classic"], min_length=1)
    population_size: PositiveInt = 32
    games_per_scenario: PositiveInt = Field(100, description="Fresh training games per scenario, every generation.")
    sigma: float = Field(0.4, gt=0)
    tournament_k: PositiveInt = 3


SNAKE, CHECKERS = frozenset({"snake"}), frozenset({"checkers"})
# Checkers and the bandit each have one interface (docs/design/0007); Snake has several to choose from.
BOARD32 = frozenset({"checkers/board32.v1+evaluate1ply.v1"})

register_algorithm(
    Algorithm("gp", GpParams, GENERATIONS, frozenset({"symbolic-regression"}), "linear GP on the fixed benchmark")
)
register_algorithm(Algorithm("neuroevolution", SnakeNeuroParams, GENERATIONS, SNAKE, "a fixed-topology policy network"))
register_algorithm(
    Algorithm(
        "neuroevolution", CheckersNeuroParams, GENERATIONS, CHECKERS, "a position evaluator, by match fitness", BOARD32
    )
)
register_algorithm(Algorithm("neat", SnakeNeatParams, GENERATIONS, SNAKE, "a policy graph whose structure evolves"))
register_algorithm(
    Algorithm("neat", CheckersNeatParams, GENERATIONS, CHECKERS, "an evaluator graph, by match fitness", BOARD32)
)
register_algorithm(
    Algorithm(
        "distill", DistillParams, GENERATIONS, CHECKERS, "an evaluator fitted to a deeper search's labels", BOARD32
    )
)
register_algorithm(
    Algorithm(
        "bandit_evolve",
        BanditEvolveParams,
        GENERATIONS,
        frozenset({"bandit"}),
        "epsilon-greedy's settings",
        frozenset({"bandit/none.v1+arm.v1"}),
    )
)
