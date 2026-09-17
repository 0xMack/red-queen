import random
from collections import Counter

from evolve import (
    Instruction,
    LinearCrossoverMutation,
    LinearProgram,
    SymbolicRegressionFitness,
    TournamentSelection,
    evolve,
    random_program,
)
from evolve.genome import op_add


def test_random_program_runs_and_produces_correct_register_count():
    rng = random.Random(0)
    program = random_program(num_instructions=8, num_registers=4, num_inputs=2, rng=rng)

    result = program.run([1.0, 2.0])

    assert len(result) == 4


def test_linear_program_output_matches_hand_traced_execution():
    # registers start at 0.0, so add(register[0], input[0]) == input[0]: a one-instruction
    # program that copies its single input straight to the output register.
    program = LinearProgram(
        instructions=(Instruction(op=0, dst=0, src_a=0, src_b=1),),  # src_b=1 -> first input
        num_registers=1,
        num_inputs=1,
        ops=(op_add,),
    )

    assert program.output([3.5]) == 3.5


def test_symbolic_regression_fitness_is_zero_for_a_perfect_fit():
    identity_program = LinearProgram(
        instructions=(Instruction(op=0, dst=0, src_a=0, src_b=1),),
        num_registers=1,
        num_inputs=1,
        ops=(op_add,),
    )
    fitness = SymbolicRegressionFitness(target=lambda x: x, inputs=[-1.0, 0.0, 1.0, 2.0])

    assert fitness.evaluate(identity_program) == 0.0


def test_symbolic_regression_fitness_penalizes_error():
    identity_program = LinearProgram(
        instructions=(Instruction(op=0, dst=0, src_a=0, src_b=1),),
        num_registers=1,
        num_inputs=1,
        ops=(op_add,),
    )
    # target is off by a constant 1.0 from what the program computes -> MSE should be exactly 1.0
    fitness = SymbolicRegressionFitness(target=lambda x: x + 1.0, inputs=[-1.0, 0.0, 1.0, 2.0])

    assert fitness.evaluate(identity_program) == -1.0


def test_tournament_selection_favors_fitter_individuals():
    population = ["a", "b", "c", "d"]
    fitnesses = [0.0, 1.0, 2.0, 3.0]  # "d" is fittest, "a" is least fit
    selection = TournamentSelection(k=2)
    rng = random.Random(42)

    counts = Counter(selection.select(population, fitnesses, rng) for _ in range(500))

    assert counts["d"] > counts["a"]


def test_linear_crossover_mutation_preserves_shape():
    rng = random.Random(1)
    a = random_program(num_instructions=10, num_registers=4, num_inputs=1, rng=rng)
    b = random_program(num_instructions=10, num_registers=4, num_inputs=1, rng=rng)

    child_no_mutation = LinearCrossoverMutation(mutation_rate=0.0).vary([a, b], rng)
    child_all_mutation = LinearCrossoverMutation(mutation_rate=1.0).vary([a, b], rng)

    for child in (child_no_mutation, child_all_mutation):
        assert len(child.instructions) == len(a.instructions)
        assert child.num_registers == a.num_registers
        assert child.num_inputs == a.num_inputs

    # with mutation_rate=0, every instruction must come from a parent, not be freshly generated
    parent_instructions = set(a.instructions) | set(b.instructions)
    assert all(instr in parent_instructions for instr in child_no_mutation.instructions)


def test_evolve_improves_best_fitness_on_a_simple_regression_problem():
    rng = random.Random(7)
    population = [
        random_program(num_instructions=12, num_registers=4, num_inputs=1, rng=rng) for _ in range(60)
    ]
    fitness = SymbolicRegressionFitness(target=lambda x: x**2, inputs=[i / 5 for i in range(-5, 6)])
    summaries = []

    evolve(
        population,
        fitness=fitness,
        selection=TournamentSelection(k=3),
        variation=LinearCrossoverMutation(mutation_rate=0.1),
        generations=60,
        on_generation=[summaries.append],
        rng=rng,
    )

    first_gen_best = summaries[0].best_fitness
    last_gen_best = summaries[-1].best_fitness
    assert last_gen_best > first_gen_best  # higher fitness = lower error = improvement
    # best_fitness is monotonically non-decreasing thanks to elitism
    assert all(
        summaries[i + 1].best_fitness >= summaries[i].best_fitness for i in range(len(summaries) - 1)
    )
