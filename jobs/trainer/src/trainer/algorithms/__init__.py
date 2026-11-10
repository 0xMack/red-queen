"""Every adapter; importing this registers them (`trainer.train` does so on first use)."""

from trainer.algorithms import (  # noqa: F401
    bandit_evolve,
    copycat,
    distill,
    gp,
    neat,
    neuroevolution,
    pbt,
    reinforcement,
    selfplay,
    tinylm,
)
