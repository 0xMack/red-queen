"""Tests for jobs/control.py's pause/resume mechanism -- exercised against a real evolve() run in a
background thread, not mocked, matching this repo's "verify by running it" norm (see
docs/CODING_GUIDELINES.md's "Lessons")."""

import random
import threading
import time

from control import make_control_callback
from evolve import (
    GenerationSummary,
    LinearCrossoverMutation,
    TournamentSelection,
    evolve,
    random_program,
)
from telemetry import SqliteRunRegistry


class SlowFitness:
    """Takes a perceptible amount of wall-clock time per generation, so a test can pause mid-run
    and actually observe progress stop, rather than the run finishing before the test can react."""

    def __init__(self, delay: float = 0.02):
        self._delay = delay

    def evaluate(self, genome) -> list[float]:
        time.sleep(self._delay)
        return [0.0]


def test_control_callback_blocks_while_paused_and_resumes(tmp_path):
    registry = SqliteRunRegistry(tmp_path / "runs.db")
    run_id = registry.create_run(config={})

    generations_seen: list[int] = []

    def track_generation(summary: GenerationSummary) -> None:
        generations_seen.append(summary.generation)

    rng = random.Random(0)
    population = [random_program(4, 2, num_inputs=1, rng=rng) for _ in range(10)]

    thread = threading.Thread(
        target=evolve,
        kwargs={
            "initial_population": population,
            "fitness": SlowFitness(),
            "selection": TournamentSelection(k=3),
            "variation": LinearCrossoverMutation(mutation_rate=0.1),
            "generations": 100,
            "on_generation": [
                track_generation,
                make_control_callback(registry, run_id, poll_interval=0.02),
            ],
            "rng": rng,
        },
        daemon=True,
    )
    thread.start()

    time.sleep(0.3)
    registry.update_status(run_id, "paused")
    # A generation already in flight when pause() is called still finishes recording (its
    # on_generation callbacks already started running before the control callback -- later in the
    # same list -- sees "paused"); settle past that one benign extra entry before taking a baseline.
    time.sleep(0.25)
    count_at_pause = len(generations_seen)
    time.sleep(0.3)
    assert len(generations_seen) == count_at_pause, "generations kept advancing while paused"

    registry.update_status(run_id, "running")
    time.sleep(0.3)
    assert len(generations_seen) > count_at_pause, "did not resume once status flipped back to running"

    # A daemon thread; not joined -- generations=100 would take ~20s to actually finish and nothing
    # here needs it to. It's killed automatically when the test process exits.
