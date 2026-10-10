"""The trainer workload (docs/design/0018): one TrainSpec in, one recorded run out. `train(spec)` validates the spec
against the algorithm's registered params (`jobcore.algorithms`) and runs its adapter (`trainer.algorithms`)."""

from trainer.registry import train

__all__ = ["train"]
