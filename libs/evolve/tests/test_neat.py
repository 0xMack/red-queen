import itertools
import json
import math
import random
from typing import ClassVar

import pytest

from evolve import (
    ConnectionGene,
    InnovationTracker,
    NeatConfig,
    NeatGenome,
    WeightVector,
    compatibility_distance,
    evolve_neat,
    initial_genome,
    network_from_json,
)
from evolve.neat import (
    add_connection,
    add_node,
    align,
    crossover,
    mutate,
    toggle_connection,
)
from evolve.networks import describe, parameter_count

CONFIG = NeatConfig()


def genome(*genes: tuple, num_inputs: int = 2, num_outputs: int = 1) -> NeatGenome:
    """(innovation, source, target, weight[, enabled]) tuples -> a genome."""
    return NeatGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        connections=tuple(ConnectionGene(*g) for g in genes),
    )


# --- genome / forward ---------------------------------------------------------------------------------


def test_node_ids_are_laid_out_inputs_then_bias_then_outputs_then_hidden():
    g = genome(num_inputs=3, num_outputs=2)

    assert (g.bias_id, g.first_output_id, g.first_hidden_id) == (3, 4, 6)


def test_forward_matches_a_hand_traced_network_with_a_hidden_node_and_bias():
    # inputs 0,1; bias 2; output 3; hidden 4.
    # x0 -> h (w=2), bias -> h (w=-1), h -> out (w=0.5), x1 -> out (w=3)
    g = genome((0, 0, 4, 2.0), (1, 2, 4, -1.0), (2, 4, 3, 0.5), (3, 1, 3, 3.0))

    hidden = math.tanh(2.0 * 0.7 + -1.0 * 1.0)
    expected = math.tanh(0.5 * hidden + 3.0 * -0.2)
    assert g.forward([0.7, -0.2]) == [pytest.approx(expected)]


def test_disabled_connections_do_not_contribute():
    g = genome((0, 0, 3, 5.0, False), (1, 1, 3, 1.0))

    assert g.forward([100.0, 0.5]) == [pytest.approx(math.tanh(0.5))]


def test_an_output_with_no_incoming_connections_is_tanh_of_zero():
    assert genome(num_outputs=2).forward([1.0, 1.0]) == [0.0, 0.0]


def test_dead_end_hidden_nodes_are_skipped_but_do_not_change_the_result():
    # hidden node 4 feeds nothing that reaches the output
    with_dead_end = genome((0, 0, 3, 1.0), (1, 0, 4, 9.0))
    without = genome((0, 0, 3, 1.0))

    assert with_dead_end.forward([0.3, 0.0]) == without.forward([0.3, 0.0])
    assert 4 not in with_dead_end.activations([0.3, 0.0])


def test_activations_reports_every_live_node():
    g = genome((0, 0, 4, 1.0), (1, 4, 3, 1.0))

    acts = g.activations([0.5, 0.0])

    assert acts[4] == pytest.approx(math.tanh(0.5))
    assert acts[3] == pytest.approx(math.tanh(math.tanh(0.5)))
    assert acts[2] == 1.0  # bias


def test_json_round_trips_and_network_from_json_dispatches_on_type():
    g = genome((0, 0, 4, 2.0), (1, 4, 3, -0.5, False))
    wv = WeightVector(weights=(1.0, 2.0), layer_sizes=(1, 1))

    assert NeatGenome.from_json(g.to_json()) == g
    assert network_from_json(g.to_json()) == g
    assert network_from_json(wv.to_json()) == wv  # no "type" field: old artifacts keep loading
    with pytest.raises(ValueError):
        network_from_json(json.dumps({"type": "nope"}))


def test_parameter_count_and_describe_cover_both_network_kinds():
    g = genome((0, 0, 4, 1.0), (1, 4, 3, 1.0), (2, 1, 3, 1.0, False))
    wv = WeightVector(weights=(0.0,) * 17, layer_sizes=(2, 4, 1))

    assert parameter_count(g) == 2  # enabled only
    assert parameter_count(wv) == 17
    assert "1 hidden" in describe(g) and describe(wv) == "2 → 4 → 1"


# --- historical markings / mutation -------------------------------------------------------------------


def test_the_same_structural_change_gets_the_same_innovation_number():
    tracker = InnovationTracker(first_hidden_id=4)

    a = tracker.connection(0, 3)
    b = tracker.connection(1, 3)

    assert a != b
    assert tracker.connection(0, 3) == a
    assert tracker.split_node(a) == tracker.split_node(a) == 4
    assert tracker.split_node(b) == 5


def test_initial_genome_wires_every_input_and_the_bias_to_every_output_with_no_hidden_nodes():
    tracker = InnovationTracker(first_hidden_id=2 + 1 + 3)
    g = initial_genome(2, 3, tracker, random.Random(0))

    assert len(g.connections) == (2 + 1) * 3
    assert g.hidden_ids == ()
    assert [c.innovation for c in g.connections] == sorted(c.innovation for c in g.connections)


def test_add_node_splits_a_connection_the_way_the_paper_describes():
    tracker = InnovationTracker(first_hidden_id=4)
    g = genome((tracker.connection(0, 3), 0, 3, 0.8))

    child = add_node(g, tracker, CONFIG, random.Random(0))

    old, into, out = sorted(child.connections, key=lambda c: c.innovation)
    assert not old.enabled and (old.source, old.target) == (0, 3)
    assert (into.source, into.target, into.weight) == (0, 4, 1.0)  # in -> new: weight 1
    assert (out.source, out.target, out.weight) == (
        4,
        3,
        0.8,
    )  # new -> out: the old weight


def test_add_node_never_splits_a_bias_connection_or_a_disabled_one():
    tracker = InnovationTracker(first_hidden_id=4)
    only_bias = genome((tracker.connection(2, 3), 2, 3, 1.0))
    only_disabled = genome((tracker.connection(0, 3), 0, 3, 1.0, False))

    assert add_node(only_bias, tracker, CONFIG, random.Random(0)) == only_bias
    assert add_node(only_disabled, tracker, CONFIG, random.Random(0)) == only_disabled


def test_add_node_does_not_burn_node_ids_on_connections_it_does_not_split():
    tracker = InnovationTracker(first_hidden_id=4)
    g = genome((tracker.connection(0, 3), 0, 3, 1.0), (tracker.connection(1, 3), 1, 3, 1.0))

    child = add_node(g, tracker, CONFIG, random.Random(0))

    assert child.hidden_ids == (4,)  # the first id, not the second: only the chosen split was allocated


def test_add_node_respects_the_hidden_node_cap():
    tracker = InnovationTracker(first_hidden_id=4)
    g = genome((tracker.connection(0, 3), 0, 3, 1.0))
    capped = NeatConfig(max_hidden_nodes=0)

    assert add_node(g, tracker, capped, random.Random(0)) == g


def test_two_genomes_splitting_the_same_connection_get_the_same_hidden_node():
    tracker = InnovationTracker(first_hidden_id=4)
    base = genome((tracker.connection(0, 3), 0, 3, 1.0))

    a = add_node(base, tracker, CONFIG, random.Random(1))
    b = add_node(base, tracker, CONFIG, random.Random(2))

    assert [c.innovation for c in a.connections] == [c.innovation for c in b.connections]
    assert a.hidden_ids == b.hidden_ids == (4,)


def _is_acyclic(g: NeatGenome) -> bool:
    successors: dict[int, list[int]] = {}
    for c in g.connections:
        successors.setdefault(c.source, []).append(c.target)
    state: dict[int, int] = {}

    def visit(n: int) -> bool:
        if state.get(n) == 1:
            return False
        if state.get(n) == 2:
            return True
        state[n] = 1
        ok = all(visit(m) for m in successors.get(n, ()))
        state[n] = 2
        return ok

    return all(visit(n) for n in list(successors))


def test_mutation_never_duplicates_a_connection_or_creates_a_cycle():
    rng = random.Random(3)
    tracker = InnovationTracker(first_hidden_id=2 + 1 + 1)
    g = initial_genome(2, 1, tracker, rng)
    config = NeatConfig(add_connection_rate=1.0, add_node_rate=1.0, toggle_rate=0.3)

    for _ in range(200):
        g = mutate(g, config, tracker, rng)
        pairs = [(c.source, c.target) for c in g.connections]
        assert len(pairs) == len(set(pairs))
        assert _is_acyclic(g)
        g.forward([0.1, -0.4])
    assert len(g.hidden_ids) > 3  # the loop really did grow structure


def test_add_connection_refuses_a_connection_that_would_close_a_loop():
    # a chain in -> h4 -> out. The only legal new gene from h4 is to the output (or nothing), never back to
    # an input or to itself.
    tracker = InnovationTracker(first_hidden_id=4)
    g = genome((0, 0, 4, 1.0), (1, 4, 3, 1.0))
    rng = random.Random(0)

    for _ in range(100):
        assert _is_acyclic(add_connection(g, tracker, CONFIG, rng))


def test_toggle_flips_exactly_one_gene():
    g = genome((0, 0, 3, 1.0), (1, 1, 3, 1.0))

    toggled = toggle_connection(g, random.Random(0))

    assert sum(a.enabled != b.enabled for a, b in zip(g.connections, toggled.connections, strict=True)) == 1


# --- alignment / crossover / compatibility -------------------------------------------------------------


# The paper's Figure 4 shape: parent A has innovations 1-5 and 8; parent B has 1-7 and 9-10.
PARENT_A = genome(*[(i, 0, 3, 1.0) for i in (1, 2, 3, 4, 5, 8)])
PARENT_B = genome(*[(i, 1, 3, -1.0) for i in (1, 2, 3, 4, 5, 6, 7, 9, 10)])


def test_align_classifies_matching_disjoint_and_excess_genes_like_the_paper():
    matching, dis_a, dis_b, exc_a, exc_b = align(PARENT_A, PARENT_B)

    assert [f.innovation for f, _ in matching] == [1, 2, 3, 4, 5]
    assert [c.innovation for c in dis_a] == [8]  # A-only, inside B's range (<= 10)
    assert [c.innovation for c in dis_b] == [6, 7]  # B-only, inside A's range (<= 8)
    assert exc_a == []
    assert [c.innovation for c in exc_b] == [9, 10]  # B-only, beyond A's max (8)


def test_crossover_takes_disjoint_and_excess_genes_only_from_the_fitter_parent():
    child = crossover(PARENT_A, PARENT_B, CONFIG, random.Random(0))
    assert [c.innovation for c in child.connections] == [1, 2, 3, 4, 5, 8]

    child = crossover(PARENT_B, PARENT_A, CONFIG, random.Random(0))
    assert [c.innovation for c in child.connections] == [1, 2, 3, 4, 5, 6, 7, 9, 10]


def test_crossover_picks_each_matching_gene_from_either_parent():
    origins = set()
    for seed in range(40):
        child = crossover(PARENT_A, PARENT_B, CONFIG, random.Random(seed))
        origins |= {c.source for c in child.connections if c.innovation <= 5}  # A's source is 0, B's is 1

    assert origins == {0, 1}


def test_a_gene_disabled_in_either_parent_is_usually_disabled_in_the_child():
    a = genome((0, 0, 3, 1.0, False))
    b = genome((0, 0, 3, 1.0, True))
    always = NeatConfig(disabled_inherit_rate=1.0)
    never = NeatConfig(disabled_inherit_rate=0.0)

    assert not crossover(a, b, always, random.Random(0)).connections[0].enabled
    assert crossover(a, b, never, random.Random(0)).connections[0].enabled


def test_compatibility_distance_is_zero_for_identical_genomes_and_symmetric():
    assert compatibility_distance(PARENT_A, PARENT_A, CONFIG) == 0.0
    assert compatibility_distance(PARENT_A, PARENT_B, CONFIG) == compatibility_distance(PARENT_B, PARENT_A, CONFIG)


def test_compatibility_distance_matches_a_hand_computed_value():
    # E=2 (9,10), D=3 (8 in A; 6,7 in B), N=1 (small genomes), mean |dw| over matching = |1-(-1)| = 2
    config = NeatConfig(excess_coefficient=1.0, disjoint_coefficient=1.0, weight_coefficient=0.4)

    assert compatibility_distance(PARENT_A, PARENT_B, config) == pytest.approx(2 + 3 + 0.4 * 2.0)


def test_compatibility_distance_normalizes_by_size_once_genomes_are_large():
    big_a = genome(*[(i, 0, 3, 0.0) for i in range(30)])
    big_b = genome(*[(i, 0, 3, 0.0) for i in range(31)])  # one excess gene, N = 31

    assert compatibility_distance(big_a, big_b, CONFIG) == pytest.approx(1 / 31)


# --- the loop -----------------------------------------------------------------------------------------


class XorFitness:
    """4 test cases, one per XOR row: 1 - squared error, so a perfect network scores 1.0 on each."""

    CASES: ClassVar[list[tuple[list[float], float]]] = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    def evaluate(self, g: NeatGenome) -> list[float]:
        # outputs are tanh in [-1, 1]; map to [0, 1] so the targets are reachable
        return [1.0 - ((g.forward(x)[0] + 1) / 2 - y) ** 2 for x, y in self.CASES]


def _run_xor(config: NeatConfig, seed: int, generations: int = 60, population_size: int = 100):
    rng = random.Random(seed)
    tracker = InnovationTracker(first_hidden_id=2 + 1 + 1)
    population = [initial_genome(2, 1, tracker, rng) for _ in range(population_size)]
    summaries = []
    evolve_neat(population, tracker, XorFitness(), config, generations, [summaries.append], rng)
    return summaries


def test_evolve_neat_learns_xor_and_reports_species_and_structure_in_extras():
    summaries = _run_xor(NeatConfig(add_node_rate=0.1, add_connection_rate=0.2), seed=0)

    # XOR is not linearly separable: the initial no-hidden-node structure cannot get above ~0.75 mean
    assert summaries[-1].best_fitness > 0.9
    assert summaries[-1].best_fitness > summaries[0].best_fitness
    assert {"species", "champion_hidden_nodes", "champion_connections"} <= summaries[-1].extras.keys()
    assert summaries[-1].champion.complexity()[0] >= 1  # it grew structure to get there


def test_turning_speciation_off_leaves_exactly_one_species():
    off = _run_xor(NeatConfig(speciation=False), seed=1, generations=15)
    on = _run_xor(NeatConfig(compatibility_threshold=0.5), seed=1, generations=15)

    assert all(s.extras["species"] == 1 for s in off)
    assert max(s.extras["species"] for s in on) > 1


def test_evolve_neat_keeps_the_population_size_constant():
    rng = random.Random(0)
    tracker = InnovationTracker(first_hidden_id=4)
    population = [initial_genome(2, 1, tracker, rng) for _ in range(37)]

    final = evolve_neat(
        population,
        tracker,
        XorFitness(),
        NeatConfig(compatibility_threshold=1.0),
        12,
        rng=rng,
    )

    assert len(final) == 37


def test_evolve_neat_never_loses_its_best_genome_between_generations():
    summaries = _run_xor(NeatConfig(), seed=2, generations=25)

    bests = [s.best_fitness for s in summaries]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in itertools.pairwise(bests))  # elitism: the champion is carried over


def test_a_target_species_count_steers_the_compatibility_threshold_toward_it():
    # a start threshold so high nothing splits: with a target, the threshold must fall until species appear
    summaries = _run_xor(
        NeatConfig(compatibility_threshold=50.0, target_species=4, threshold_step=2.0),
        seed=0,
        generations=40,
    )

    thresholds = [s.extras["compatibility_threshold"] for s in summaries]
    assert thresholds[0] == 50.0 and thresholds[-1] < 30.0
    assert max(s.extras["species"] for s in summaries) > 1
