import math
import random

from evolve import (
    GaussianMutation,
    LexicaseSelection,
    ParetoSelection,
    SimulationFitnessEvaluator,
    TournamentSelection,
    WeightVector,
    evolve,
    random_weight_vector,
)
from evolve.neuro import _forward, _param_count


def test_param_count_matches_a_hand_computed_network_shape():
    # layer (2,4,1): (2*4 weights + 4 biases) + (4*1 weights + 1 bias) = 12 + 5 = 17
    assert _param_count((2, 4, 1)) == 17


def test_forward_matches_a_hand_traced_single_neuron_network():
    # 1 input -> 1 output, weight=2.0, bias=0.5: tanh(2.0 * x + 0.5)
    weights = (2.0, 0.5)
    result = _forward(weights, layer_sizes=(1, 1), observation=[1.0])

    assert result == [math.tanh(2.0 * 1.0 + 0.5)]


def test_weight_vector_act_is_bounded_by_tanh():
    rng = random.Random(0)
    genome = random_weight_vector((2, 4, 1), rng, scale=10.0)  # large scale to stress-test bounds

    for observation in [(-5.0, 5.0), (0.0, 0.0), (3.0, -3.0)]:
        assert -1.0 <= genome.act(observation) <= 1.0


def test_l2_norm_matches_a_hand_computed_value():
    genome = WeightVector(weights=(3.0, 4.0), layer_sizes=(1, 1))

    assert genome.l2_norm() == 5.0  # 3-4-5 triangle


def test_to_json_from_json_round_trips():
    genome = WeightVector(weights=(1.0, -2.5, 0.3), layer_sizes=(2, 1))

    restored = WeightVector.from_json(genome.to_json())

    assert restored == genome
    assert isinstance(restored.weights, tuple)  # not a list -- JSON has no tuple type
    assert isinstance(restored.layer_sizes, tuple)


def test_random_weight_vector_has_the_right_number_of_parameters():
    rng = random.Random(0)
    genome = random_weight_vector((2, 4, 1), rng)

    assert len(genome.weights) == 17


def test_gaussian_mutation_preserves_shape_and_changes_weights():
    rng = random.Random(0)
    parent = random_weight_vector((2, 4, 1), rng)

    child = GaussianMutation(sigma=0.5).vary([parent], rng)

    assert len(child.weights) == len(parent.weights)
    assert child.layer_sizes == parent.layer_sizes
    assert child.weights != parent.weights


class _ConstantTargetEnv:
    """Minimal Environment for fast tests: one step, reward is -(action - target)**2."""

    def __init__(self, target: float):
        self._target = target

    def reset(self):
        return (0.0, 0.0)

    def step(self, action):
        return (0.0, 0.0), -((action - self._target) ** 2), True


def test_simulation_fitness_evaluator_returns_one_value_per_environment():
    envs = [_ConstantTargetEnv(0.3), _ConstantTargetEnv(-0.3)]
    genome = WeightVector(weights=tuple([0.0] * _param_count((2, 4, 1))), layer_sizes=(2, 4, 1))
    fitness = SimulationFitnessEvaluator(envs=envs, act=lambda g, obs: g.act(obs), max_steps=10)

    result = fitness.evaluate(genome)

    assert len(result) == 2


def test_simulation_fitness_evaluator_stops_episode_on_done():
    always_done_env = _ConstantTargetEnv(0.0)
    genome = WeightVector(weights=tuple([0.0] * _param_count((2, 4, 1))), layer_sizes=(2, 4, 1))
    fitness = SimulationFitnessEvaluator(envs=[always_done_env], act=lambda g, obs: g.act(obs), max_steps=1000)

    # if this didn't stop on done=True, evaluate() would run 1000 steps instead of 1 -- both
    # finish "instantly" in wall-clock terms here, so assert on the actual mechanism instead:
    # reset() then a single step() should account for the whole returned fitness.
    result = fitness.evaluate(genome)
    observation = always_done_env.reset()
    action = genome.act(observation)
    _obs, expected_reward, _done = always_done_env.step(action)

    assert result == [expected_reward]


def test_simulation_fitness_evaluator_counts_episodes_and_steps():
    # Training-cost counters (docs/design/0007): each env here ends after exactly one step.
    genome = WeightVector(weights=tuple([0.0] * _param_count((2, 4, 1))), layer_sizes=(2, 4, 1))
    fitness = SimulationFitnessEvaluator(envs=_two_case_envs(), act=lambda g, obs: g.act(obs), max_steps=50)

    fitness.evaluate(genome)
    fitness.evaluate(genome)

    assert (fitness.episodes, fitness.steps) == (4, 4)


def _two_case_envs():
    return [_ConstantTargetEnv(0.6), _ConstantTargetEnv(-0.6)]


def test_evolve_improves_with_weight_vector_and_tournament_selection():
    rng = random.Random(0)
    population = [random_weight_vector((2, 4, 1), rng, scale=0.5) for _ in range(30)]
    fitness = SimulationFitnessEvaluator(envs=_two_case_envs(), act=lambda g, obs: g.act(obs), max_steps=1)
    summaries = []

    evolve(
        population,
        fitness=fitness,
        selection=TournamentSelection(k=3),
        variation=GaussianMutation(sigma=0.2),
        generations=30,
        on_generation=[summaries.append],
        rng=rng,
    )

    assert summaries[-1].best_fitness > summaries[0].best_fitness


def test_evolve_improves_with_weight_vector_and_lexicase_selection():
    rng = random.Random(0)
    population = [random_weight_vector((2, 4, 1), rng, scale=0.5) for _ in range(30)]
    fitness = SimulationFitnessEvaluator(envs=_two_case_envs(), act=lambda g, obs: g.act(obs), max_steps=1)
    summaries = []

    evolve(
        population,
        fitness=fitness,
        selection=LexicaseSelection(),
        variation=GaussianMutation(sigma=0.2),
        generations=30,
        on_generation=[summaries.append],
        rng=rng,
    )

    assert summaries[-1].best_fitness > summaries[0].best_fitness


def test_evolve_runs_with_weight_vector_and_pareto_selection():
    rng = random.Random(0)
    population = [random_weight_vector((2, 4, 1), rng, scale=0.5) for _ in range(30)]
    fitness = SimulationFitnessEvaluator(envs=_two_case_envs(), act=lambda g, obs: g.act(obs), max_steps=1)
    selection = ParetoSelection(complexity=lambda g: g.l2_norm(), k=3)
    summaries = []

    evolve(
        population,
        fitness=fitness,
        selection=selection,
        variation=GaussianMutation(sigma=0.2),
        generations=30,
        on_generation=[summaries.append],
        rng=rng,
    )

    assert summaries[-1].best_fitness > summaries[0].best_fitness


def test_simulation_fitness_evaluator_can_swap_environments():
    genome = WeightVector(weights=tuple([0.0] * _param_count((2, 4, 1))), layer_sizes=(2, 4, 1))
    fitness = SimulationFitnessEvaluator(envs=_two_case_envs(), act=lambda g, obs: g.act(obs), max_steps=5)
    assert len(fitness.evaluate(genome)) == 2

    fitness.set_environments([_ConstantTargetEnv(0.1), _ConstantTargetEnv(0.2), _ConstantTargetEnv(0.3)])

    assert len(fitness.evaluate(genome)) == 3


def test_gaussian_mutation_rate_perturbs_only_a_fraction_of_the_weights():
    import random

    from evolve import GaussianMutation, WeightVector

    parent = WeightVector(weights=tuple([0.5] * 400), layer_sizes=(1, 1))  # layer_sizes unused by the mutation
    everything = GaussianMutation(sigma=0.1).vary([parent], random.Random(0))
    sparse = GaussianMutation(sigma=0.1, rate=0.05).vary([parent], random.Random(0))

    changed = lambda child: sum(1 for a, b in zip(parent.weights, child.weights, strict=True) if a != b)
    assert changed(everything) == 400
    assert 5 <= changed(sparse) <= 40  # ~5% of 400
