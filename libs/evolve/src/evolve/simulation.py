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
# (genome, environments, max_steps) -> one (total reward, steps taken) per environment, each episode played
# from a fresh reset -- a whole generation's episodes run by whatever can run them faster than the Python loop.
Rollout = Callable[[Any, Sequence["Environment"], int], list[tuple[float, int]]]


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
    SymbolicRegressionFitness's `run` -- this evaluator never touches genome internals. `rollout`, if
    given, plays the episodes instead (e.g. the game core running the whole episode natively, see
    jobs/snake_neuro_run.py's `make_rollout`); it must return exactly what this loop would.

    `episodes`/`steps` count everything this evaluator has simulated -- exact, hardware-independent
    training-cost counters (docs/design/0007), cheap enough to always keep.
    """

    def __init__(
        self,
        envs: Sequence[Environment],
        act: Callable[[Genome, Observation], Action],
        max_steps: int = 200,
        rollout: Rollout | None = None,
    ):
        self._envs = list(envs)
        self._act = act
        self._max_steps = max_steps
        self._rollout = rollout
        self.episodes = 0
        self.steps = 0

    def set_environments(self, envs: Sequence[Environment]) -> None:
        """Swap the test cases, e.g. fresh seeds every generation so a population can't memorize a
        fixed handful of games (docs/design/0007). Takes effect from the next evaluate() call."""
        self._envs = list(envs)

    def evaluate(self, genome: Genome) -> list[float]:
        if self._rollout is not None:
            results = self._rollout(genome, self._envs, self._max_steps)
            self.episodes += len(results)
            self.steps += sum(steps for _, steps in results)
            return [total for total, _ in results]
        fitnesses = []
        for env in self._envs:
            observation = env.reset()
            total_reward = 0.0
            steps = 0
            for _ in range(self._max_steps):
                action = self._act(genome, observation)
                observation, reward, done = env.step(action)
                total_reward += reward
                steps += 1
                if done:
                    break
            fitnesses.append(total_reward)
            self.episodes += 1
            self.steps += steps
        return fitnesses
