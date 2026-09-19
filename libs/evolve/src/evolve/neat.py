"""NEAT (NeuroEvolution of Augmenting Topologies, Stanley & Miikkulainen 2002) -- the second
neuroevolution representation, and the first whose *structure* evolves (docs/design/0008).

`neuro.WeightVector` fixes the network's shape up front and evolves only the numbers. A
`NeatGenome` is a *graph*: a list of connection genes, each tagged with a global **innovation
number**, and evolution can add connections and add nodes (by splitting a connection) as well as
perturb weights. Three ideas make that workable, all here:

1. **Historical markings** (`InnovationTracker`): the same structural change gets the same
   innovation number wherever it arises, so two genomes with different structures can still be
   lined up gene-by-gene for crossover (`crossover`) and compared (`compatibility_distance`).
2. **Speciation**: genomes are grouped by structural similarity and compete mostly inside their
   own species (fitness sharing), so a new structure -- usually worse until its weights are tuned
   -- isn't killed by the incumbent topology before it has a chance.
3. **Complexification**: the initial population has *no hidden nodes* (inputs wired straight to
   outputs); structure is added only when it pays for itself.

Differences from the paper, on purpose:
- Feedforward only: `add_connection` refuses anything that would create a cycle, so there is no
  recurrent state and a network is a pure function of its observation (same as `WeightVector`).
- tanh everywhere (hidden and output) instead of the steepened sigmoid, for parity with
  `neuro._forward` and bounded outputs.
- Fitness comes from any `FitnessEvaluator` (a list of per-case values, reduced to the mean) --
  NEAT's own selection is species-based, so lexicase has no role here.

Pure stdlib, like the rest of this package, so `apps/frontend`'s Pyodide bridge loads a trained
genome back with the exact same code that evolved it.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property

from evolve.fitness import FitnessEvaluator
from evolve.population import GenerationCallback, GenerationSummary

# --- Genome ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConnectionGene:
    innovation: (
        int  # global historical marking: same (source, target) -> same number, forever
    )
    source: int  # node id
    target: int  # node id
    weight: float
    enabled: bool = True


@dataclass(frozen=True)
class NeatGenome:
    """A feedforward network as a list of connection genes.

    Node ids are fixed by role: inputs are `0..num_inputs-1`, the bias node (always outputs 1.0) is
    `num_inputs`, outputs follow, and every id from `first_hidden_id` up is a hidden node handed
    out by the `InnovationTracker`. Nodes are implied by the connections (a hidden node exists iff
    some gene touches it), so there is no separate node list to keep consistent.

    Not `slots=True`: `_plan` (the compiled evaluation order) is a `cached_property`, built once per
    genome -- a genome is evaluated thousands of times per generation, mutated once.
    """

    num_inputs: int
    num_outputs: int
    connections: tuple[ConnectionGene, ...]  # sorted by innovation number

    @property
    def bias_id(self) -> int:
        return self.num_inputs

    @property
    def first_output_id(self) -> int:
        return self.num_inputs + 1

    @property
    def first_hidden_id(self) -> int:
        return self.num_inputs + 1 + self.num_outputs

    @property
    def hidden_ids(self) -> tuple[int, ...]:
        ids = {
            n
            for c in self.connections
            for n in (c.source, c.target)
            if n >= self.first_hidden_id
        }
        return tuple(sorted(ids))

    @property
    def enabled_connections(self) -> tuple[ConnectionGene, ...]:
        return tuple(c for c in self.connections if c.enabled)

    @cached_property
    def _plan(
        self,
    ) -> tuple[list[int], list[tuple[int, list[tuple[int, float]]]], list[int]]:
        """(node id of every value slot, [(slot, [(source_slot, weight), ...]) in topological
        order], output slots). Only nodes that can reach an output are kept -- dead-end hidden
        nodes cost nothing at inference time."""
        incoming: dict[int, list[tuple[int, float]]] = {}
        for c in self.connections:
            if c.enabled:
                incoming.setdefault(c.target, []).append((c.source, c.weight))

        outputs = list(
            range(self.first_output_id, self.first_output_id + self.num_outputs)
        )
        needed: set[int] = set(outputs)
        stack = list(outputs)
        while stack:
            for source, _ in incoming.get(stack.pop(), ()):
                if source >= self.first_hidden_id and source not in needed:
                    needed.add(source)
                    stack.append(source)

        # Kahn's algorithm over the needed hidden/output nodes; inputs/bias are always ready.
        remaining = {
            n: sum(1 for s, _ in incoming.get(n, ()) if s in needed) for n in needed
        }
        dependents: dict[int, list[int]] = {}
        for n in needed:
            for s, _ in incoming.get(n, ()):
                if s in needed:
                    dependents.setdefault(s, []).append(n)
        ready = sorted(n for n, k in remaining.items() if k == 0)
        order: list[int] = []
        while ready:
            n = ready.pop()
            order.append(n)
            for d in dependents.get(n, ()):
                remaining[d] -= 1
                if remaining[d] == 0:
                    ready.append(d)

        # value slots: inputs, bias, then every needed node in evaluation order
        node_of_slot = list(range(self.num_inputs + 1)) + order
        slot = {node: i for i, node in enumerate(node_of_slot)}
        steps = [
            (slot[n], [(slot[s], w) for s, w in incoming.get(n, ())]) for n in order
        ]
        return node_of_slot, steps, [slot[o] for o in outputs]

    def _run(self, observation: Sequence[float]) -> list[float]:
        node_of_slot, steps, _ = self._plan
        values = [0.0] * len(node_of_slot)
        values[: self.num_inputs] = observation
        values[self.num_inputs] = 1.0  # bias
        for slot, incoming in steps:
            total = 0.0
            for source, weight in incoming:
                total += weight * values[source]
            values[slot] = math.tanh(total)
        return values

    def forward(self, observation: Sequence[float]) -> list[float]:
        values = self._run(observation)
        return [values[s] for s in self._plan[2]]

    def activations(self, observation: Sequence[float]) -> dict[int, float]:
        """Every live node's value for `observation` (node id -> activation), for visualization."""
        values = self._run(observation)
        return {node: values[slot] for slot, node in enumerate(self._plan[0])}

    def act(self, observation: Sequence[float]) -> float:
        return self.forward(observation)[0]

    def complexity(self) -> tuple[int, int]:
        """(hidden nodes, enabled connections) -- how big the evolved structure has become."""
        return len(self.hidden_ids), len(self.enabled_connections)

    def to_json(self) -> str:
        """Round-trippable wire format (`type: "neat"`, see `networks.network_from_json`)."""
        return json.dumps(
            {
                "type": "neat",
                "num_inputs": self.num_inputs,
                "num_outputs": self.num_outputs,
                "connections": [
                    {
                        "innovation": c.innovation,
                        "source": c.source,
                        "target": c.target,
                        "weight": c.weight,
                        "enabled": c.enabled,
                    }
                    for c in self.connections
                ],
            }
        )

    @staticmethod
    def from_json(text: str) -> NeatGenome:
        data = json.loads(text)
        return NeatGenome(
            num_inputs=data["num_inputs"],
            num_outputs=data["num_outputs"],
            connections=tuple(
                ConnectionGene(
                    innovation=c["innovation"],
                    source=c["source"],
                    target=c["target"],
                    weight=c["weight"],
                    enabled=c["enabled"],
                )
                for c in data["connections"]
            ),
        )


# --- Historical markings -----------------------------------------------------------------------------


class InnovationTracker:
    """Hands out innovation numbers and hidden-node ids, remembering what it already gave out.

    Asking for the same `(source, target)` connection twice returns the same innovation number, and
    splitting the same connection twice returns the same new node id -- that is the *only* reason
    two independently-evolved genomes can be aligned later. One tracker per run; it is mutable
    state shared by every mutation, so it lives outside the (immutable) genomes.
    """

    def __init__(self, first_hidden_id: int):
        self._connection: dict[tuple[int, int], int] = {}
        self._split_node: dict[int, int] = {}
        self._next_innovation = 0
        self._next_node = first_hidden_id

    def connection(self, source: int, target: int) -> int:
        key = (source, target)
        if key not in self._connection:
            self._connection[key] = self._next_innovation
            self._next_innovation += 1
        return self._connection[key]

    def split_node(self, connection_innovation: int) -> int:
        if connection_innovation not in self._split_node:
            self._split_node[connection_innovation] = self._next_node
            self._next_node += 1
        return self._split_node[connection_innovation]

    def known_split_node(self, connection_innovation: int) -> int | None:
        """The node id a split of this connection got, if it was ever split -- without handing one out."""
        return self._split_node.get(connection_innovation)

    @property
    def innovations(self) -> int:
        return self._next_innovation


def initial_genome(
    num_inputs: int,
    num_outputs: int,
    tracker: InnovationTracker,
    rng: random.Random,
    weight_scale: float = 1.0,
) -> NeatGenome:
    """The minimal starting structure: every input (and the bias) wired straight to every output,
    random weights, no hidden nodes."""
    connections = [
        ConnectionGene(
            innovation=tracker.connection(source, num_inputs + 1 + o),
            source=source,
            target=num_inputs + 1 + o,
            weight=rng.uniform(-weight_scale, weight_scale),
        )
        for o in range(num_outputs)
        for source in range(num_inputs + 1)
    ]
    return NeatGenome(
        num_inputs, num_outputs, tuple(sorted(connections, key=lambda c: c.innovation))
    )


# --- Configuration -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class NeatConfig:
    """Every NEAT knob in one place. Defaults follow the paper's spirit, adapted to a 100-genome
    Snake population; each value that a run overrides is recorded in that run's config."""

    # variation
    weight_mutation_rate: float = 0.8  # chance a genome's weights are mutated at all
    weight_perturb_sigma: float = 0.2  # Gaussian step for a perturbed weight
    weight_replace_rate: float = (
        0.1  # of the mutated weights, fraction replaced outright
    )
    weight_replace_scale: float = 1.0
    add_connection_rate: float = 0.08
    add_node_rate: float = 0.04
    toggle_rate: float = 0.01
    crossover_rate: float = 0.75  # else the child is a mutated clone of one parent
    interspecies_rate: float = 0.001
    disabled_inherit_rate: float = (
        0.75  # child of a gene disabled in either parent: chance it stays disabled
    )
    # speciation
    compatibility_threshold: float = 3.0
    # The paper's fixed threshold assumes small genomes: distance is normalized by gene count once a
    # genome has >= 20 genes, so for a bigger starting network (Snake: 36) a fixed 3.0 is never
    # reached and everything stays one species. Setting a target makes the threshold itself adapt --
    # nudged down when there are too few species, up when too many -- starting from the value above.
    target_species: int | None = None
    threshold_step: float = 0.1
    min_threshold: float = 0.05
    excess_coefficient: float = 1.0  # c1
    disjoint_coefficient: float = 1.0  # c2
    weight_coefficient: float = 0.4  # c3
    # reproduction
    survival_threshold: float = 0.2  # top fraction of each species allowed to breed
    stagnation_limit: int = (
        15  # generations without species improvement before it stops breeding
    )
    min_species_kept: int = 2  # stagnant species are only culled down to this many
    species_elitism_size: int = 5  # a species this large keeps its champion unchanged
    # sizing
    initial_weight_scale: float = 1.0
    max_hidden_nodes: int = (
        40  # add_node stops here, a safety valve against runaway growth
    )
    speciation: bool = (
        True  # False: one big species -- the ablation that shows what speciation buys
    )


# --- Variation ---------------------------------------------------------------------------------------


def _creates_cycle(genome: NeatGenome, source: int, target: int) -> bool:
    """Would adding source -> target close a loop? True iff `source` is already reachable from
    `target` (over every gene, enabled or not, so re-enabling a gene can never create one either)."""
    if source == target:
        return True
    successors: dict[int, list[int]] = {}
    for c in genome.connections:
        successors.setdefault(c.source, []).append(c.target)
    stack, seen = [target], {target}
    while stack:
        node = stack.pop()
        if node == source:
            return True
        for nxt in successors.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def mutate_weights(
    genome: NeatGenome, config: NeatConfig, rng: random.Random
) -> NeatGenome:
    new = []
    for c in genome.connections:
        if rng.random() < config.weight_replace_rate:
            weight = rng.uniform(
                -config.weight_replace_scale, config.weight_replace_scale
            )
        else:
            weight = c.weight + rng.gauss(0.0, config.weight_perturb_sigma)
        new.append(replace(c, weight=weight))
    return replace(genome, connections=tuple(new))


def add_connection(
    genome: NeatGenome,
    tracker: InnovationTracker,
    config: NeatConfig,
    rng: random.Random,
) -> NeatGenome:
    """Connect two previously-unconnected nodes (source: input/bias/hidden, target: hidden/output),
    keeping the graph acyclic. A no-op if no legal pair is found after a few tries."""
    sources = [*range(genome.first_output_id), *genome.hidden_ids]
    targets = [
        *range(genome.first_output_id, genome.first_hidden_id),
        *genome.hidden_ids,
    ]
    existing = {(c.source, c.target) for c in genome.connections}
    for _ in range(20):
        source, target = rng.choice(sources), rng.choice(targets)
        if (source, target) in existing or _creates_cycle(genome, source, target):
            continue
        gene = ConnectionGene(
            innovation=tracker.connection(source, target),
            source=source,
            target=target,
            weight=rng.uniform(
                -config.weight_replace_scale, config.weight_replace_scale
            ),
        )
        return replace(
            genome,
            connections=tuple(
                sorted([*genome.connections, gene], key=lambda c: c.innovation)
            ),
        )
    return genome


def add_node(
    genome: NeatGenome,
    tracker: InnovationTracker,
    config: NeatConfig,
    rng: random.Random,
) -> NeatGenome:
    """Split an enabled connection A -> B into A -> N -> B: the old gene is disabled, A -> N gets
    weight 1.0 and N -> B inherits the old weight, so the new node starts out (almost) neutral --
    the network's behaviour barely changes, and the new structure is free to be tuned from there."""
    if len(genome.hidden_ids) >= config.max_hidden_nodes:
        return genome
    present = set(genome.hidden_ids)
    # A connection is splittable unless this genome already contains the node that splitting it
    # would (re)create -- possible if a disabled gene got re-enabled. Asking the tracker must not
    # allocate ids for connections we then don't split.
    candidates = [
        c
        for c in genome.connections
        if c.enabled
        and c.source != genome.bias_id
        and tracker.known_split_node(c.innovation) not in present
    ]
    if not candidates:
        return genome
    old = rng.choice(candidates)
    node = tracker.split_node(old.innovation)
    incoming = ConnectionGene(
        tracker.connection(old.source, node), old.source, node, 1.0
    )
    outgoing = ConnectionGene(
        tracker.connection(node, old.target), node, old.target, old.weight
    )
    kept = [
        replace(c, enabled=False) if c.innovation == old.innovation else c
        for c in genome.connections
    ]
    return replace(
        genome,
        connections=tuple(
            sorted([*kept, incoming, outgoing], key=lambda c: c.innovation)
        ),
    )


def toggle_connection(genome: NeatGenome, rng: random.Random) -> NeatGenome:
    index = rng.randrange(len(genome.connections))
    new = list(genome.connections)
    new[index] = replace(new[index], enabled=not new[index].enabled)
    return replace(genome, connections=tuple(new))


def mutate(
    genome: NeatGenome,
    config: NeatConfig,
    tracker: InnovationTracker,
    rng: random.Random,
) -> NeatGenome:
    if rng.random() < config.weight_mutation_rate:
        genome = mutate_weights(genome, config, rng)
    if rng.random() < config.add_connection_rate:
        genome = add_connection(genome, tracker, config, rng)
    if rng.random() < config.add_node_rate:
        genome = add_node(genome, tracker, config, rng)
    if rng.random() < config.toggle_rate:
        genome = toggle_connection(genome, rng)
    return genome


def align(
    a: NeatGenome, b: NeatGenome
) -> tuple[
    list[tuple[ConnectionGene, ConnectionGene]],
    list[ConnectionGene],
    list[ConnectionGene],
    list[ConnectionGene],
    list[ConnectionGene],
]:
    """Line two genomes up by innovation number.

    Returns (matching pairs, disjoint-in-a, disjoint-in-b, excess-in-a, excess-in-b). *Matching*
    genes exist in both; *disjoint* genes are in only one and fall inside the other's innovation
    range; *excess* genes are in only one and lie beyond the other's highest innovation number.
    """
    by_a = {c.innovation: c for c in a.connections}
    by_b = {c.innovation: c for c in b.connections}
    max_a = max(by_a, default=-1)
    max_b = max(by_b, default=-1)
    matching = [(by_a[i], by_b[i]) for i in sorted(by_a.keys() & by_b.keys())]
    only_a = [by_a[i] for i in sorted(by_a.keys() - by_b.keys())]
    only_b = [by_b[i] for i in sorted(by_b.keys() - by_a.keys())]
    return (
        matching,
        [c for c in only_a if c.innovation <= max_b],
        [c for c in only_b if c.innovation <= max_a],
        [c for c in only_a if c.innovation > max_b],
        [c for c in only_b if c.innovation > max_a],
    )


def crossover(
    fitter: NeatGenome, other: NeatGenome, config: NeatConfig, rng: random.Random
) -> NeatGenome:
    """Offspring of two parents (`fitter` first). Matching genes come from either parent at random;
    disjoint and excess genes only from the fitter one. A gene disabled in either parent stays
    disabled with probability `disabled_inherit_rate`.

    The child's genes are always a subset of the fitter parent's, so an acyclic fitter parent
    guarantees an acyclic child -- no cycle check needed here.
    """
    matching, *_ = align(fitter, other)
    matched = {f.innovation: (f, o) for f, o in matching}
    child = []
    for gene in fitter.connections:
        pair = matched.get(gene.innovation)
        if pair is None:
            child.append(gene)
            continue
        f, o = pair
        chosen = f if rng.random() < 0.5 else o
        enabled = chosen.enabled
        if not (f.enabled and o.enabled):
            enabled = rng.random() >= config.disabled_inherit_rate
        child.append(replace(chosen, enabled=enabled))
    return replace(fitter, connections=tuple(child))


# --- Speciation --------------------------------------------------------------------------------------


def compatibility_distance(a: NeatGenome, b: NeatGenome, config: NeatConfig) -> float:
    """delta = c1*E/N + c2*D/N + c3*W: structural difference (excess E, disjoint D genes, normalized
    by the larger genome's gene count N -- 1 for small genomes) plus the mean absolute weight
    difference W over matching genes."""
    matching, dis_a, dis_b, exc_a, exc_b = align(a, b)
    n = max(len(a.connections), len(b.connections))
    n = 1 if n < 20 else n
    weight_diff = (
        statistics.fmean(abs(x.weight - y.weight) for x, y in matching)
        if matching
        else 0.0
    )
    return (
        config.excess_coefficient * (len(exc_a) + len(exc_b)) / n
        + config.disjoint_coefficient * (len(dis_a) + len(dis_b)) / n
        + config.weight_coefficient * weight_diff
    )


@dataclass
class Species:
    species_id: int
    representative: NeatGenome
    members: list[int] = field(
        default_factory=list
    )  # indices into the current population
    best_fitness: float = -math.inf
    last_improved: int = 0  # generation


def _speciate(
    population: list[NeatGenome],
    species: list[Species],
    config: NeatConfig,
    generation: int,
    next_species_id: int,
) -> tuple[list[Species], int]:
    for s in species:
        s.members = []
    if not config.speciation:
        if not species:
            species = [Species(next_species_id, population[0])]
            next_species_id += 1
        species = species[:1]
        species[0].members = list(range(len(population)))
        return species, next_species_id
    for index, genome in enumerate(population):
        for s in species:
            if (
                compatibility_distance(genome, s.representative, config)
                < config.compatibility_threshold
            ):
                s.members.append(index)
                break
        else:
            species.append(
                Species(next_species_id, genome, [index], last_improved=generation)
            )
            next_species_id += 1
    return [s for s in species if s.members], next_species_id


def _allocate(shares: list[float], total: int) -> list[int]:
    """Split `total` offspring across species in proportion to `shares` (largest remainder)."""
    weight = sum(shares)
    if weight <= 0:
        base = [total // len(shares)] * len(shares)
    else:
        exact = [total * s / weight for s in shares]
        base = [int(x) for x in exact]
        remainders = sorted(
            range(len(shares)), key=lambda i: exact[i] - base[i], reverse=True
        )
        for i in remainders[: total - sum(base)]:
            base[i] += 1
    for i in range(total - sum(base)):
        base[i % len(base)] += 1
    return base


# --- The loop ----------------------------------------------------------------------------------------


def evolve_neat(
    initial_population: list[NeatGenome],
    tracker: InnovationTracker,
    fitness: FitnessEvaluator,
    config: NeatConfig,
    generations: int,
    on_generation: Iterable[GenerationCallback] = (),
    rng: random.Random | None = None,
) -> list[NeatGenome]:
    """Evaluate -> report -> speciate -> reproduce, for `generations` rounds.

    Reports through the same `GenerationSummary` the generic `evolve()` uses (so every callback in
    `jobs/` works unchanged), with NEAT-specific numbers in `summary.extras`: species count and the
    champion's structure size.
    """
    rng = rng or random.Random()
    population = list(initial_population)
    callbacks = list(on_generation)
    species: list[Species] = []
    next_species_id = 0
    threshold = config.compatibility_threshold

    for generation in range(generations):
        case_fitnesses = [fitness.evaluate(genome) for genome in population]
        aggregate = [statistics.fmean(cf) for cf in case_fitnesses]
        ranked = sorted(
            range(len(population)), key=lambda i: aggregate[i], reverse=True
        )
        champion = population[ranked[0]]

        species, next_species_id = _speciate(
            population,
            species,
            replace(config, compatibility_threshold=threshold),
            generation,
            next_species_id,
        )
        for s in species:
            best = max(aggregate[i] for i in s.members)
            if best > s.best_fitness:
                s.best_fitness, s.last_improved = best, generation

        hidden, connections = champion.complexity()
        summary: GenerationSummary[NeatGenome] = GenerationSummary(
            generation=generation,
            best_fitness=aggregate[ranked[0]],
            mean_fitness=statistics.fmean(aggregate),
            worst_fitness=aggregate[ranked[-1]],
            diversity=statistics.pstdev(aggregate) if len(aggregate) > 1 else 0.0,
            champion=champion,
            extras={
                "species": float(len(species)),
                "champion_hidden_nodes": float(hidden),
                "champion_connections": float(connections),
                "mean_connections": statistics.fmean(
                    len(g.enabled_connections) for g in population
                ),
                "innovations": float(tracker.innovations),
                "compatibility_threshold": threshold,
            },
        )
        for callback in callbacks:
            callback(summary)

        if config.target_species is not None and config.speciation:
            if len(species) < config.target_species:
                threshold = max(config.min_threshold, threshold - config.threshold_step)
            elif len(species) > config.target_species:
                threshold += config.threshold_step
        # next generation's genomes are compared against a random member of each species
        for s in species:
            s.representative = population[rng.choice(s.members)]
        population = _reproduce(
            population, aggregate, species, config, tracker, generation, rng
        )
    return population


def _reproduce(
    population: list[NeatGenome],
    aggregate: list[float],
    species: list[Species],
    config: NeatConfig,
    tracker: InnovationTracker,
    generation: int,
    rng: random.Random,
) -> list[NeatGenome]:
    size = len(population)
    floor = min(aggregate)
    shifted = [
        f - floor + 1e-6 for f in aggregate
    ]  # fitness can be negative; sharing needs > 0

    # stagnant species stop breeding (but never cull below `min_species_kept`)
    by_best = sorted(species, key=lambda s: s.best_fitness, reverse=True)
    protected = {s.species_id for s in by_best[: config.min_species_kept]}
    breeding = [
        s
        for s in species
        if generation - s.last_improved < config.stagnation_limit
        or s.species_id in protected
    ]

    # fitness sharing: each member's fitness is divided by its species' size, so a species' total
    # claim on the next generation is its *mean* shifted fitness, not its headcount
    shares = [statistics.fmean(shifted[i] for i in s.members) for s in breeding]
    quotas = _allocate(shares, size)

    global_champion = max(range(size), key=lambda i: aggregate[i])
    next_population: list[NeatGenome] = []
    for s, quota in zip(breeding, quotas, strict=True):
        if quota == 0:
            continue
        members = sorted(s.members, key=lambda i: aggregate[i], reverse=True)
        if len(members) >= config.species_elitism_size or global_champion in members:
            next_population.append(
                population[members[0]]
            )  # champion carried over unchanged
            quota -= 1
        parents = members[: max(1, math.ceil(len(members) * config.survival_threshold))]
        for _ in range(quota):
            a = rng.choice(parents)
            if len(parents) > 1 and rng.random() < config.crossover_rate:
                b = rng.choice(parents)
                if rng.random() < config.interspecies_rate and len(breeding) > 1:
                    other = rng.choice([x for x in breeding if x is not s])
                    b = rng.choice(other.members)
                fitter, weaker = (a, b) if aggregate[a] >= aggregate[b] else (b, a)
                child = crossover(population[fitter], population[weaker], config, rng)
            else:
                child = population[a]
            next_population.append(mutate(child, config, tracker, rng))

    # rounding / culled species can leave the population short: top up from the overall best
    ranked = sorted(range(size), key=lambda i: aggregate[i], reverse=True)
    while len(next_population) < size:
        parent = population[rng.choice(ranked[: max(1, size // 5)])]
        next_population.append(mutate(parent, config, tracker, rng))
    return next_population[:size]


__all__ = [
    "ConnectionGene",
    "InnovationTracker",
    "NeatConfig",
    "NeatGenome",
    "Species",
    "add_connection",
    "add_node",
    "align",
    "compatibility_distance",
    "crossover",
    "evolve_neat",
    "initial_genome",
    "mutate",
    "mutate_weights",
    "toggle_connection",
]
