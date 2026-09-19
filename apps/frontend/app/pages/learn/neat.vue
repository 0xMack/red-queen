<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
const genomeCode = `@dataclass(frozen=True, slots=True)
class ConnectionGene:
    innovation: int   # global historical marking: same (source, target) -> same number, forever
    source: int       # node id
    target: int
    weight: float
    enabled: bool = True

@dataclass(frozen=True)
class NeatGenome:
    num_inputs: int
    num_outputs: int
    connections: tuple[ConnectionGene, ...]   # nodes are implied by what the genes touch`

const addNodeCode = `def add_node(genome, tracker, config, rng):
    old = rng.choice(splittable_connections)            # an enabled A -> B
    node = tracker.split_node(old.innovation)           # same split -> same node id, everywhere
    into = ConnectionGene(tracker.connection(old.source, node), old.source, node, 1.0)
    out  = ConnectionGene(tracker.connection(node, old.target), node, old.target, old.weight)
    # old gene disabled; the new node starts out nearly neutral
    return replace(genome, connections=sorted([*disable(old), into, out]))`

const trackerCode = `class InnovationTracker:
    def connection(self, source, target):
        key = (source, target)
        if key not in self._connection:            # first time anyone has made this connection
            self._connection[key] = self._next_innovation
            self._next_innovation += 1
        return self._connection[key]               # ...and the same number ever after`

const crossoverCode = `def crossover(fitter, other, config, rng):
    matched = {f.innovation: (f, o) for f, o in align(fitter, other).matching}
    child = []
    for gene in fitter.connections:                # only ever the fitter parent's genes
        pair = matched.get(gene.innovation)
        if pair is None:
            child.append(gene)                     # disjoint / excess: inherited from the fitter
            continue
        f, o = pair
        chosen = f if rng.random() < 0.5 else o    # matching: a coin flip
        enabled = chosen.enabled
        if not (f.enabled and o.enabled):          # disabled in either parent...
            enabled = rng.random() >= config.disabled_inherit_rate   # ...usually stays disabled
        child.append(replace(chosen, enabled=enabled))
    return replace(fitter, connections=tuple(child))`

const distanceCode = `def compatibility_distance(a, b, config):
    matching, dis_a, dis_b, exc_a, exc_b = align(a, b)
    n = max(len(a.connections), len(b.connections))
    n = 1 if n < 20 else n                          # don't normalize small genomes
    weight_diff = mean(abs(x.weight - y.weight) for x, y in matching)
    return (config.excess_coefficient   * (len(exc_a) + len(exc_b)) / n
          + config.disjoint_coefficient * (len(dis_a) + len(dis_b)) / n
          + config.weight_coefficient   * weight_diff)`

const sharingCode = `# fitness can be negative in Snake, so shift it before dividing it up
shifted = [f - min(fitness) + 1e-6 for f in fitness]
# each species' claim on the next generation = its MEAN shifted fitness (fitness sharing)
shares  = [mean(shifted[i] for i in species.members) for species in breeding]
quotas  = allocate(shares, population_size)          # offspring per species, largest remainder`

const adaptiveCode = `# A fixed threshold assumes small genomes. Snake starts at 36 genes, where distance is
# divided by N >= 20 -- so structural differences barely register and one species swallows everything.
# Aim for a species COUNT instead, and let the threshold chase it:
if len(species) < config.target_species:
    threshold -= config.threshold_step
elif len(species) > config.target_species:
    threshold += config.threshold_step`
</script>

<template>
  <article class="prose-chapter">
    <p>
      <NuxtLink to="/learn/neuroevolution">Neuroevolution</NuxtLink> left one decision outside the loop: the network's shape. Someone picked 16
      hidden units, and evolution can only ever adjust the numbers inside that choice. <strong>NEAT</strong> -- NeuroEvolution of Augmenting
      Topologies (Stanley &amp; Miikkulainen, 2002) -- puts the shape inside the loop too. A NEAT network starts as small as a network can be and
      <em>grows</em>, and the interesting part is the three ideas needed to make that work without falling apart.
    </p>

    <Callout variant="note" title="Two things evolve, and they don't mix well">
      Structure and weights are entangled: new structure usually doesn't help until the weights around it have been tuned to use it, so
      it tends to look <em>no better, or worse,</em> than the network it grew from -- and a population that only rewards current fitness will
      delete it before that tuning happens. Every idea below is a fix for some part of that.
    </Callout>

    <h2>A genome is a list of genes</h2>
    <p>
      Instead of a grid of weights, a NEAT genome is a list of <strong>connection genes</strong>: each says "node A feeds node B with this weight,
      and it's currently on or off". Nodes aren't stored at all -- a hidden node exists if some gene touches it. The starting genome has no hidden
      nodes: every input, plus a constant bias, wired straight to every output.
    </p>
    <CodeBlock lang="python" :code="genomeCode" />
    <p>
      There are four mutations. Two change weights or on/off state; two change <em>structure</em>: <strong>add connection</strong> joins two nodes that
      weren't linked (never in a way that makes a loop -- these are feed-forward networks), and <strong>add node</strong> splits an existing
      connection in two. Grow one below, and watch the table: every structural change adds rows, and every row carries a number in the first column.
    </p>
    <NeatGenomeExplorer />

    <h2>Adding a node without breaking anything</h2>
    <p>
      Splitting a connection A → B into A → N → B needs a rule for the two new weights, and NEAT's is chosen to make the change nearly free:
      the incoming half gets weight <strong>1</strong> and the outgoing half keeps the <strong>old weight</strong>, while the old gene is switched off.
      The new node passes A's signal through almost unchanged, so the network still behaves about as well as before -- with a new place for later
      mutations to work. (Try it above: click <em>Add node</em> and compare the fitness before and after.)
    </p>
    <CodeBlock lang="python" :code="addNodeCode" />
    <p>
      The old gene is disabled, not deleted. It stays in the genome as a record -- and, as the next sections show, that record matters.
    </p>

    <h2>Innovation numbers: telling genes apart</h2>
    <p>
      Here is the problem from the last chapter again. Two networks that grew different structures have genomes of different lengths and shapes;
      there is no "gene <em>i</em> in each" to swap. NEAT's answer is to give every structural change a permanent identity the moment it first
      appears. A global counter hands out an <strong>innovation number</strong> to each new (source → target) connection, and hands out the
      <em>same</em> number if any genome, anywhere in the run, makes that same connection again:
    </p>
    <CodeBlock lang="python" :code="trackerCode" />
    <p>
      Now two genomes can be lined up by innovation number, gene for gene. Genes both have are <strong>matching</strong>; genes only one has are
      <strong>disjoint</strong> if they fall inside the other's range of numbers and <strong>excess</strong> if they lie beyond it. Crossover uses
      exactly that alignment: matching genes come from either parent at random, everything else only from the fitter parent, and a gene that's
      disabled in either parent usually stays disabled. Flip which parent is fitter below and see the child change:
    </p>
    <NeatCrossoverDemo />
    <CodeBlock lang="python" :code="crossoverCode" />
    <p>
      Notice what this does <em>not</em> do. The child never gains a gene neither parent had, so an acyclic fitter parent guarantees an acyclic
      child, with no cycle check needed. Crossover recombines structure that already exists; only the mutations invent new structure.
    </p>

    <h2>Speciation: protecting new structure</h2>
    <p>
      A new node usually scores worse than the tuned network it came from. Left in one big population, it loses every tournament and vanishes. NEAT
      shelters it by grouping genomes into <strong>species</strong> by structural similarity, and making genomes compete mostly <em>inside</em>
      their own species. The similarity measure reuses the alignment from crossover -- how many excess and disjoint genes two genomes have, and how
      far apart their matching weights are:
    </p>
    <CodeBlock lang="python" :code="distanceCode" />
    <SpeciationDemo />
    <p>
      Genomes within a threshold δₜ of a species' representative join it; otherwise they found a new species. Then <strong>fitness sharing</strong>
      does the protecting: a species' claim on the next generation is its <em>average</em> fitness, not its headcount, so a large species can't
      crowd out a small one just by being large, and a promising newcomer gets offspring in proportion to how good it is, not how many there are of it.
    </p>
    <CodeBlock lang="python" :code="sharingCode" />
    <p>
      A species that hasn't improved for 15 generations stops breeding (the best two are always kept), and a species of five or more keeps its
      champion unchanged, so progress can't be lost to bad luck.
    </p>

    <h2>Putting it together, live</h2>
    <p>
      This is the whole algorithm running on XOR, starting from genomes with <strong>no hidden nodes</strong> -- which can score at most 0.75, so
      any better result means evolution grew structure to get it. Press Play and watch the fitness curve step up when a useful hidden node is
      found, the species bars split and merge, and the champion's graph appear.
    </p>
    <NeatLab />
    <p>
      The experiment at the bottom is the reason speciation is in the algorithm. It runs the identical loop eight times each way, changing one thing.
      Runs that fail get stuck at exactly 0.75 -- the ceiling for a network with no hidden nodes, which is also what outputting a constant 0.5
      scores. That's the situation speciation is designed for: a new hidden node starts out worse than the incumbents sitting on that plateau, so
      with no protected niche to grow in, it's bred out before its weights can be tuned.
    </p>

    <h2>NEAT against Snake</h2>
    <p>
      The same algorithm, unchanged, on the game from
      <NuxtLink to="/learn/teaching-a-snake">the Snake case study</NuxtLink>: 11 hand-engineered inputs, 3 outputs (turn left, straight, turn right),
      genomes that begin as 36 direct connections and nothing else. Below is a finished run's champion playing live (the actual Snake and the
      actual <code>NeatGenome</code>, in your browser), with its evolved graph lighting up move by move.
    </p>
    <NeatSnakeChampion />
    <p>
      That's one champion. A single run is an anecdote -- a different random seed can move a Snake policy's score by several points -- so the
      project ran a tracked experiment: four algorithm variants, <strong>five seeds each</strong>, an identical budget, every run recorded to
      telemetry and scored on the same 200 unseen games (<code>jobs/snake_experiment.py</code>, docs/design/0008):
    </p>
    <SnakeNeatResults />
    <ul>
      <li>
        <strong>NEAT beat the existing neuroevolution setup.</strong> Its five runs scored 19.2–21.1 against 14.0–17.6 for lexicase-selected fixed
        networks (+3.8 on average; exact permutation test p = 0.008), with about a quarter of the parameters and a quarter of the training time.
        Every NEAT run also beat the 3-line greedy heuristic (18.5); no lexicase run did.
      </li>
      <li>
        <strong>The fairer comparison is smaller, and inconclusive.</strong> NEAT selects on mean fitness, and lexicase looks weaker than tournament for
        the fixed network here (1.8 points, p = 0.14 -- also not established), so the arm to compare against is the tournament-selected one: NEAT is still ahead, by 2.0 points, but at five
        seeds that is suggestive, not established (p = 0.056), and the ranges overlap -- the best tournament run (20.5) matches a typical NEAT one.
      </li>
      <li>
        <strong>Speciation bought reliability, and even that is one data point.</strong> Without it, four runs scored 14.2–21.4 (median 20.2, about
        the same as NEAT's 20.6) and one never took off (1.5: its best training fitness never rose above 2.2 in 250 generations). That collapse drags the mean down by 4.6 points, but a single
        failure in five runs can't be told apart from chance (p = 0.35).
      </li>
    </ul>
    <Callout variant="warning" title="What this doesn't show">
      One game, one board size, five seeds per arm, and hyperparameters tuned for neither side (NEAT's mutation rates and species target are first-guess defaults; the
      fixed network's σ = 0.2 and 16 hidden units are the project's earlier choices). It supports "NEAT is a competitive, cheap way to train a Snake
      policy here", not "NEAT beats neuroevolution".
    </Callout>
    <p>
      One Snake-specific detail is worth knowing because nothing errors when it's wrong. The paper's compatibility threshold (3.0) assumes small
      genomes: once a genome has 20 or more genes, distance is divided by the gene count, and Snake's 36-gene starting genomes barely register as
      different from each other. In the first trial run the species count sat at 1 in every generation, which would have made NEAT silently
      equivalent to "neuroevolution with a growing network". The fix is to steer the threshold toward a target species count instead of fixing it:
    </p>
    <CodeBlock lang="python" :code="adaptiveCode" />
    <Callout variant="finding" title="Found by running it: species count 1, every generation">
      The first Snake trial run reported <code>species = 1</code> for every generation it recorded, with a threshold that never came into play.
      Nothing was wrong with the code -- the unit tests passed -- the configuration just didn't suit a 36-gene genome. It only showed up in the per-generation
      <code>species</code> curve the run records, which is why <code>evolve_neat</code> reports it (and the champion's structure size) through
      telemetry's <code>extras</code>: an algorithm with a hidden internal mechanism needs that mechanism on a chart.
    </Callout>

    <h2>What NEAT costs</h2>
    <p>
      NEAT's networks aren't minimal, either. The live XOR lab's speciated champions carry about six hidden nodes for a problem that needs one, and the
      Snake champions 6–11. Growing structure is cheap for it, and nothing in the loop pays for tidiness.
    </p>
    <p>
      NEAT has more moving parts than a fixed-shape network -- innovation tracking, a distance function, species bookkeeping, a dozen rates -- and
      each of them is something that can be configured badly without anything visibly breaking, as the species-count bug above shows. Its networks
      are also irregular graphs rather than dense layers, so they don't map onto matrix hardware the way a fixed MLP does. What you buy is that
      nobody has to choose the architecture, the search starts in a small space and only widens if it pays, and the evolved network is small enough
      to read. Whether that trade is worth it depends on the problem; the section above is this project's measurement of it on one, not a general
      answer.
    </p>
  </article>
</template>
