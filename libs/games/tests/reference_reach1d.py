"""The oracle for the Rust port of ReachTarget1D (docs/design/0009 Decision 2): `games/reach1d.py` exactly as it
was in pure Python before the game core moved to Rust, unchanged. `test_native_parity.py` holds the
Rust to it. Test-only on purpose -- the executable spec, not a second implementation to run.

The original module docstring follows.
"""

"""ReachTarget1D: a toy 1D continuous-control task.

An agent with (position, velocity) must reach and stay near a fixed target by choosing an
acceleration each step. Implements evolve.simulation.Environment's reset()/step() shape
(docs/design/0002) -- this package never imports evolve; the dependency points the other way, from
a SimulationFitnessEvaluator to this environment, same shape as evolve/telemetry.
"""


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
        self.position = start_position
        self.velocity = start_velocity

    def reset(self) -> tuple[float, float]:
        self.position = self.start_position
        self.velocity = self.start_velocity
        return self._observation()

    def step(self, action: float) -> tuple[tuple[float, float], float, bool]:
        """`action` is clamped to [-1, 1] and scaled by `max_acceleration`."""
        acceleration = max(-1.0, min(1.0, action)) * self.max_acceleration
        self.velocity = (self.velocity + acceleration * self.dt) * self.damping
        self.position += self.velocity * self.dt
        reward = -abs(self.position - self.target)
        return self._observation(), reward, False

    def _observation(self) -> tuple[float, float]:
        # Position relative to the target, not raw position: two episodes with the same start
        # position but different targets are otherwise observationally identical at reset (same
        # (position, velocity)) despite needing opposite actions -- no policy can solve both from
        # that observation, no matter how it's evolved. Found by actually running neuroevolution
        # against this environment and noticing two benchmark scenarios could never both improve.
        return (self.position - self.target, self.velocity)


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
