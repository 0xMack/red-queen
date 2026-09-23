"""Reinforcement learning from scratch (docs/design/0010).

The learning happens in the Rust core (`libs/rl/rust/core`, exposed here as `rl._native`): environments, networks
with backprop, optimizers, agents and the training loop. This package is its Python face for training jobs --
`Trainer` advances one training iteration per call, so Python never sits in the inner loop.
"""

from rl._native import ALGORITHMS, Trainer, env_info, evaluate_baseline

__all__ = ["ALGORITHMS", "Trainer", "env_info", "evaluate_baseline"]
