"""Reinforcement learning (docs/design/0010): the single-agent `rl.Trainer` algorithms, Checkers self-play (TD(λ) /
TD-Leaf, AlphaZero-style) and population-based training over self-play learners (docs/design/0014).

Each learner's own settings (`alpha`, `n_step`, `double`, `clip`, `hidden`, `search_depth`, `pool_every`, ...) are
the Rust core's, passed through as `agent` and recorded as the run's `params` -- the core rejects a name it doesn't
know. What sits beside `agent` is the workload's: how training is chunked, monitored and recorded.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from jobcore.algorithms.evolution import BOARD32, CHECKERS
from jobcore.specs import Algorithm, register_algorithm

# libs/rl's `rl.ALGORITHMS`, named here so validating a spec doesn't import the native extension.
RL_ALGORITHMS = ("random", "q_learning", "sarsa", "dqn", "reinforce", "a2c", "ppo")
# Snake under any of its interfaces (tabular agents need one with a discretizer), or Reach1D (continuous control).
RL_GAMES = frozenset({"snake", "reach1d"})
ITERATIONS = frozenset({"iterations"})


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # int | float, not float: a setting is recorded in the run config exactly as given (`hidden: 8`, not 8.0).
    agent: dict[str, int | float] = Field(
        default_factory=dict, description="The learner's own settings (the Rust core's)."
    )


class RlParams(_Params):
    steps_per_iteration: PositiveInt = Field(10_000, description="Environment steps per recorded iteration.")
    reward: Literal["shaped", "sparse"] = "shaped"
    snapshot_every: PositiveInt = Field(1, description="Store the policy every N iterations (and the last).")


class SelfPlayParams(_Params):
    games_per_iteration: PositiveInt = 1000
    depth: PositiveInt = Field(3, description="Plies the network searches when monitored (and on the leaderboard).")
    monitor_games: PositiveInt = Field(10, description="Held-out games per opponent, seats alternating.")


class AlphaZeroParams(SelfPlayParams):
    mcts_games: PositiveInt = Field(20, description="Games the search-and-policy player plays at held-out points.")


class PbtParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    members: PositiveInt = 8
    interval: PositiveInt = Field(10_000, description="Self-play games per member between round robins.")
    depth: PositiveInt = 3
    openings_per_pair: PositiveInt = Field(2, description="Ballot openings per pair in each round robin.")
    exploit: bool = Field(True, description="False: the control -- the same population, nothing copied.")
    base: dict[str, int | float] = Field(
        {"hidden": 64, "hidden_layers": 2, "pool_every": 5000, "pool_size": 10},
        description="Every member's fixed learner settings; the evolving ones are sampled per member.",
    )
    monitor_games: PositiveInt = 10


for name in RL_ALGORITHMS:
    register_algorithm(Algorithm(f"rl.{name}", RlParams, ITERATIONS, RL_GAMES, f"libs/rl's `{name}`"))
register_algorithm(
    Algorithm("td_lambda", SelfPlayParams, ITERATIONS, CHECKERS, "a position evaluator by TD(λ) self-play", BOARD32)
)
register_algorithm(
    Algorithm(
        "alphazero", AlphaZeroParams, ITERATIONS, CHECKERS, "AlphaZero-style self-play (docs/design/0017)", BOARD32
    )
)
register_algorithm(
    Algorithm(
        "pbt", PbtParams, frozenset({"games"}), CHECKERS, "population-based training of self-play learners", BOARD32
    )
)
