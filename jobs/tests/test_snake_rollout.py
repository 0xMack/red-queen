"""The game core's native Snake rollout (Snake.play) must score a genome exactly as the Python loop does --
same moves, same total reward, same step count -- or switching training to it would silently change results."""

import random

import pytest
from evolve import NeatConfig, SimulationFitnessEvaluator, initial_genome, random_weight_vector
from evolve.neat import InnovationTracker, mutate
from evolve.networks import compiled
from games import interfaces
from games.nets import native_policy
from snake_neuro_run import BOARD, make_act, make_rollout

INTERFACES = ["snake/features.v1+relative3.v1", "snake/grid-flat.v1+relative3.v1", "snake/egocentric.v1+relative3.v1"]
SEEDS = range(40)


def _layered(n_inputs: int, rng: random.Random):
    return random_weight_vector((n_inputs, 8, 3), rng, scale=1.5)


def _grown_neat(n_inputs: int, rng: random.Random):
    """A NEAT genome with hidden nodes and extra connections -- a real graph, not the initial 2-layer one."""
    tracker = InnovationTracker(first_hidden_id=n_inputs + 1 + 3)
    config = NeatConfig(add_node_rate=0.6, add_connection_rate=0.8)
    genome = initial_genome(n_inputs, 3, tracker, rng, 1.5)
    for _ in range(12):
        genome = mutate(genome, config, tracker, rng)
    assert genome.complexity()[0] > 0
    return genome


def _evaluators(interface_id: str):
    interface = interfaces.get(interface_id)
    envs = [interface.make_game(seed=seed, **BOARD) for seed in SEEDS]
    python = SimulationFitnessEvaluator(envs=envs, act=make_act(interface), max_steps=200)
    native = SimulationFitnessEvaluator(envs=envs, act=make_act(interface), max_steps=200, rollout=make_rollout(interface))
    return interface, envs, python, native


@pytest.mark.parametrize("interface_id", INTERFACES)
@pytest.mark.parametrize("build", [_layered, _grown_neat], ids=["layered", "neat"])
def test_native_rollout_scores_exactly_like_the_python_loop(interface_id, build):
    interface, envs, python, native = _evaluators(interface_id)
    n_inputs = len(interface.observer.feature_names(envs[0]))
    rng = random.Random(7)
    for _ in range(4):
        genome = build(n_inputs, rng)
        assert native.evaluate(genome) == python.evaluate(genome)
    assert (native.episodes, native.steps) == (python.episodes, python.steps)
    assert native.steps > native.episodes  # the policies actually played, not died on step one


def test_native_policy_forward_matches_both_python_networks():
    rng = random.Random(3)
    for genome in (_layered(11, rng), _grown_neat(11, rng)):
        policy = native_policy(compiled(genome))
        for _ in range(50):
            observation = [rng.uniform(-1, 1) for _ in range(11)]
            assert policy.forward(observation) == genome.forward(observation)


def test_a_policy_of_the_wrong_shape_is_refused():
    interface = interfaces.get("snake/grid-flat.v1+relative3.v1")
    game = interface.make_game(seed=0, **BOARD)
    wrong = native_policy(compiled(_layered(11, random.Random(0))))
    with pytest.raises(ValueError, match="100 inputs"):
        game.play(wrong, 50)
    with pytest.raises(ValueError, match="unknown network kind"):
        native_policy({"kind": "tree"})
