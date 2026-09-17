"""Simulation-based fitness -- the counterpart to fitness.py's dataset-based
SymbolicRegressionFitness (docs/design/0002/0003).

`Environment` is the interface this module depends on; a concrete game/simulation (e.g.
games.reach1d in libs/games) implements it but never imports from here -- the dependency points
from the evaluator to the interface, not from a game to the algorithm (same shape as
evolve/telemetry in docs/design/0001).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

Genome = TypeVar("Genome")
Observation = Any
Action = Any


class Environment(Protocol):
    def reset(self) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Returns (observation, reward, done)."""
        ...


class SimulationFitnessEvaluator:
    """FitnessEvaluator over a fixed set of environments/episodes -- one fitness value (total
    episode reward) per environment, matching the per-test-case contract every other
    FitnessEvaluator in this package follows (fitness.py), so LexicaseSelection works on
    simulation fitness exactly as it does on dataset fitness, with no changes needed. Each
    environment is a different "test case": a different start state, a different target, a
    different scenario the policy has to handle.

    `act` is injected `(genome, observation) -> action`, the same pattern as
    SymbolicRegressionFitness's `run` -- this evaluator never touches genome internals.
    """

    def __init__(
        self,
        envs: Sequence[Environment],
        act: Callable[[Genome, Observation], Action],
        max_steps: int = 200,
    ):
        self._envs = list(envs)
        self._act = act
        self._max_steps = max_steps

    def evaluate(self, genome: Genome) -> list[float]:
        fitnesses = []
        for env in self._envs:
            observation = env.reset()
            total_reward = 0.0
            for _ in range(self._max_steps):
                action = self._act(genome, observation)
                observation, reward, done = env.step(action)
                total_reward += reward
                if done:
                    break
            fitnesses.append(total_reward)
        return fitnesses
