import random

from evolve import (
    FunctionNode,
    LexicaseSelection,
    ParetoSelection,
    SymbolicRegressionFitness,
    Terminal,
    TournamentSelection,
    TreeCrossoverMutation,
    TreeProgram,
    evolve,
    random_tree_program,
)
from evolve.tree import _flatten


def test_terminal_variable_evaluates_to_the_input():
    assert Terminal(value=None).evaluate(3.5) == 3.5


def test_terminal_constant_evaluates_to_itself_regardless_of_input():
    assert Terminal(value=2.0).evaluate(99.0) == 2.0


def test_function_node_evaluates_hand_traced():
    # op=0 is op_add in DEFAULT_OPS -- x + 3.0
    tree = TreeProgram(root=FunctionNode(op=0, left=Terminal(value=None), right=Terminal(value=3.0)))

    assert tree.evaluate(2.0) == 5.0


def test_node_count_and_depth_for_a_hand_built_tree():
    # root
    #  |-- x                         (depth 0)
    #  `-- (op=2)                    (depth 1)
    #       |-- 1.0                  (depth 0)
    #       `-- x                    (depth 0)
    right = FunctionNode(op=2, left=Terminal(value=1.0), right=Terminal(value=None))
    tree = TreeProgram(root=FunctionNode(op=0, left=Terminal(value=None), right=right))

    assert tree.node_count() == 5
    assert tree.depth() == 2


def test_random_tree_program_respects_max_depth():
    rng = random.Random(0)
    for _ in range(50):
        tree = random_tree_program(max_depth=4, rng=rng)
        assert tree.depth() <= 4


def test_random_tree_program_evaluates_without_error_across_inputs():
    rng = random.Random(1)
    tree = random_tree_program(max_depth=5, rng=rng)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        result = tree.evaluate(x)
        assert isinstance(result, float)


def test_tree_crossover_mutation_respects_max_tree_depth():
    rng = random.Random(2)
    variation = TreeCrossoverMutation(mutation_rate=0.3, max_tree_depth=3)
    a = random_tree_program(max_depth=3, rng=rng)
    b = random_tree_program(max_depth=3, rng=rng)

    for _ in range(100):
        child = variation.vary([a, b], rng)
        assert child.depth() <= 3


def test_tree_crossover_without_mutation_only_uses_parent_material():
    rng = random.Random(4)
    a = random_tree_program(max_depth=4, rng=rng)
    b = random_tree_program(max_depth=4, rng=rng)
    variation = TreeCrossoverMutation(mutation_rate=0.0, max_tree_depth=10)

    child = variation.vary([a, b], rng)

    parent_nodes = set(_flatten(a.root)) | set(_flatten(b.root))
    assert all(node in parent_nodes for node in _flatten(child.root))


def _tree_fitness():
    return SymbolicRegressionFitness(
        target=lambda x: x**4 - 3 * x**2 + 2,
        inputs=[i / 5 for i in range(-5, 6)],
        run=lambda t, x: t.evaluate(x),
    )


def test_evolve_improves_with_tree_program_and_tournament_selection():
    rng = random.Random(0)
    population = [random_tree_program(max_depth=4, rng=rng) for _ in range(60)]
    summaries = []

    evolve(
        population,
        fitness=_tree_fitness(),
        selection=TournamentSelection(k=3),
        variation=TreeCrossoverMutation(mutation_rate=0.1),
        generations=60,
        on_generation=[summaries.append],
        rng=rng,
    )

    assert summaries[-1].best_fitness > summaries[0].best_fitness


def test_evolve_improves_with_tree_program_and_lexicase_selection():
    rng = random.Random(0)
    population = [random_tree_program(max_depth=4, rng=rng) for _ in range(60)]
    summaries = []

    evolve(
        population,
        fitness=_tree_fitness(),
        selection=LexicaseSelection(),
        variation=TreeCrossoverMutation(mutation_rate=0.1),
        generations=60,
        on_generation=[summaries.append],
        rng=rng,
    )

    assert summaries[-1].best_fitness > summaries[0].best_fitness


def test_evolve_improves_with_tree_program_and_pareto_selection():
    rng = random.Random(0)
    population = [random_tree_program(max_depth=4, rng=rng) for _ in range(60)]
    summaries = []

    evolve(
        population,
        fitness=_tree_fitness(),
        selection=ParetoSelection(complexity=lambda t: t.node_count(), k=3),
        variation=TreeCrossoverMutation(mutation_rate=0.1),
        generations=60,
        on_generation=[summaries.append],
        rng=rng,
    )

    assert summaries[-1].best_fitness > summaries[0].best_fitness
