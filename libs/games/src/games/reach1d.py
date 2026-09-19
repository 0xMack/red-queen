"""ReachTarget1D: a toy 1D continuous-control task.

An agent with (position, velocity) must reach and stay near a fixed target by choosing an
acceleration each step. Implements evolve.simulation.Environment's reset()/step() shape
(docs/design/0002) -- this package never imports evolve; the dependency points the other way, from
a SimulationFitnessEvaluator to this environment, same shape as evolve/telemetry.

The dynamics run in the Rust game core (rust/core/src/reach1d.rs, docs/design/0009); this is its
Python face.
"""

from __future__ import annotations

import copy

from games import _native


class ReachTarget1D:
    def __init__(
        self,
        target: float = 5.0,
        start_position: float = 0.0,
        start_velocity: float = 0.0,
        dt: float = 0.1,
        max_acceleration: float = 1.0,
        damping: float = 0.98,
    ):
        self.target = target
        self.start_position = start_position
        self.start_velocity = start_velocity
        self.dt = dt
        self.max_acceleration = max_acceleration
        self.damping = damping
        self._core = _native.Reach1DCore(target, start_position, start_velocity, dt, max_acceleration, damping)

    def __copy__(self):
        # The state lives in the native core, so even a "shallow" copy must not share it.
        return copy.deepcopy(self)

    @property
    def position(self) -> float:
        return self._core.position

    @position.setter
    def position(self, value: float) -> None:
        self._core.position = value

    @property
    def velocity(self) -> float:
        return self._core.velocity

    @velocity.setter
    def velocity(self, value: float) -> None:
        self._core.velocity = value

    def reset(self) -> tuple[float, float]:
        return self._core.reset()

    def step(self, action: float) -> tuple[tuple[float, float], float, bool]:
        """`action` is clamped to [-1, 1] and scaled by `max_acceleration`.

        Observation is position *relative to the target*, not raw position: two episodes with the same
        start position but different targets are otherwise observationally identical at reset despite
        needing opposite actions -- no policy can solve both from that observation, no matter how it's
        evolved. Found by actually running neuroevolution against this environment."""
        observation, reward = self._core.step(float(action))
        return observation, reward, False


def benchmark_environments() -> list[ReachTarget1D]:
    """A small fixed set of scenarios (docs/design/0003 "fixed benchmark problems as an anchor"):
    reaching from either side, a short hop, a long reach, and starting close but moving fast away
    from the target -- so a policy that only learned "always accelerate right" doesn't do well
    across all of them."""
    return [
        ReachTarget1D(target=5.0, start_position=0.0),
        ReachTarget1D(target=-5.0, start_position=0.0),
        ReachTarget1D(target=3.0, start_position=-3.0),
        ReachTarget1D(target=0.0, start_position=8.0),
        ReachTarget1D(target=2.0, start_position=2.0, start_velocity=3.0),
    ]
