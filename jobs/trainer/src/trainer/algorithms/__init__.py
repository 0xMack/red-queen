"""Every adapter; importing this registers them (`trainer.train` does so on first use)."""

from trainer.algorithms import (  # noqa: F401
    bandit_evolve,
    distill,
    gp,
    neat,
    neuroevolution,
    pbt,
    reinforcement,
    selfplay,
    tinylm,
)
