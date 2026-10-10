"""Every adapter; importing this registers them (`trainer.train` does so on first use)."""

from trainer.algorithms import bandit_evolve, distill, gp, neat, neuroevolution  # noqa: F401
