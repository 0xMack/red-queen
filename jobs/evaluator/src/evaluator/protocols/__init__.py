"""Every protocol runner; importing this registers them (`evaluator.evaluate` does so on first use)."""

from evaluator.protocols import (  # noqa: F401
    bandit_skill,
    checkers_sprt,
    checkers_versus,
    experiments,
    snake_score,
    static_round_robin,
)
