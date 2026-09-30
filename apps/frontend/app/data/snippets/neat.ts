import type { Snippet } from "~/types/code"

// NEAT's snippets. Python is libs/evolve/src/evolve/neat.py (trimmed); NEAT trains only in Python -- a champion
// reaches Rust already compiled to an evaluation plan (GraphNet), never as a genome -- so the Rust is a translation.
const SOURCE = "libs/evolve/src/evolve/neat.py"

export const genome: Snippet = {
  pseudo: `ConnectionGene:
    innovation   // a global historical marking: the same (source, target) gets the same number, forever
    source, target
    weight
    enabled

NeatGenome:
    num_inputs, num_outputs
    connections  // a list of genes; the nodes are implied by what the genes touch`,
  python: {
    source: SOURCE,
    code: `@dataclass(frozen=True, slots=True)
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
    connections: tuple[ConnectionGene, ...]   # nodes are implied by what the genes touch`,
  },
  rust: `#[derive(Clone, Copy)]
pub struct ConnectionGene {
    pub innovation: u32, // global historical marking: same (source, target) -> same number, forever
    pub source: u32,     // node id
    pub target: u32,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Clone)]
pub struct NeatGenome {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub connections: Vec<ConnectionGene>, // nodes are implied by what the genes touch
}`,
}

export const addNode: Snippet = {
  pseudo: `function add_node(genome, tracker)
    old ← a random enabled connection A → B
    N ← tracker.split_node(old)                        // the same split gets the same node id, everywhere
    disable old
    add A → N with weight 1                            // so N starts out nearly neutral:
    add N → B with old's weight                        // the network's behaviour barely changes
    return genome`,
  python: {
    source: SOURCE,
    code: `def add_node(genome, tracker, config, rng):
    old = rng.choice(splittable_connections)            # an enabled A -> B
    node = tracker.split_node(old.innovation)           # same split -> same node id, everywhere
    into = ConnectionGene(tracker.connection(old.source, node), old.source, node, 1.0)
    out  = ConnectionGene(tracker.connection(node, old.target), node, old.target, old.weight)
    # old gene disabled; the new node starts out nearly neutral
    return replace(genome, connections=sorted([*disable(old), into, out]))`,
  },
  rust: `pub fn add_node(genome: &NeatGenome, tracker: &mut InnovationTracker, rng: &mut impl Rng) -> NeatGenome {
    let old = *splittable_connections(genome, tracker).choose(rng).unwrap(); // an enabled A -> B
    let node = tracker.split_node(old.innovation); // same split -> same node id, everywhere
    let into = ConnectionGene { innovation: tracker.connection(old.source, node), target: node, weight: 1.0, ..old };
    let out = ConnectionGene { innovation: tracker.connection(node, old.target), source: node, ..old };
    // old gene disabled; the new node starts out nearly neutral
    let mut connections = genome.connections.clone();
    connections.iter_mut().filter(|c| c.innovation == old.innovation).for_each(|c| c.enabled = false);
    connections.extend([into, out]);
    connections.sort_by_key(|c| c.innovation);
    NeatGenome { connections, ..genome.clone() }
}`,
}

export const tracker: Snippet = {
  pseudo: `function innovation(source, target)
    if nobody has made the connection source → target before
        remember it under the next unused innovation number
    return the number remembered for source → target      // ...the same one ever after`,
  python: {
    source: SOURCE,
    code: `class InnovationTracker:
    def connection(self, source, target):
        key = (source, target)
        if key not in self._connection:            # first time anyone has made this connection
            self._connection[key] = self._next_innovation
            self._next_innovation += 1
        return self._connection[key]               # ...and the same number ever after`,
  },
  rust: `pub struct InnovationTracker {
    connection: HashMap<(u32, u32), u32>,
    next_innovation: u32,
}

impl InnovationTracker {
    pub fn connection(&mut self, source: u32, target: u32) -> u32 {
        let next = &mut self.next_innovation;
        *self.connection.entry((source, target)).or_insert_with(|| {
            *next += 1; // first time anyone has made this connection
            *next - 1
        }) // ...and the same number ever after
    }
}`,
}

export const crossover: Snippet = {
  pseudo: `function crossover(fitter, other)
    child ← empty
    for each gene g in fitter                          // only ever the fitter parent's genes
        if other has a gene with g's innovation number (a matching gene)
            chosen ← g or other's gene, by a coin flip
            if either copy is disabled then usually keep it disabled
        else                                           // disjoint or excess
            chosen ← g
        append chosen to child
    return child`,
  python: {
    source: SOURCE,
    code: `def crossover(fitter, other, config, rng):
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
    return replace(fitter, connections=tuple(child))`,
  },
  rust: `pub fn crossover(fitter: &NeatGenome, other: &NeatGenome, config: &NeatConfig, rng: &mut impl Rng) -> NeatGenome {
    let matched: HashMap<u32, &ConnectionGene> = other.connections.iter().map(|g| (g.innovation, g)).collect();
    let connections = fitter
        .connections
        .iter() // only ever the fitter parent's genes
        .map(|f| match matched.get(&f.innovation) {
            None => *f, // disjoint / excess: inherited from the fitter
            Some(o) => {
                let mut chosen = if rng.gen::<f64>() < 0.5 { *f } else { **o }; // matching: a coin flip
                if !(f.enabled && o.enabled) {
                    // disabled in either parent... usually stays disabled
                    chosen.enabled = rng.gen::<f64>() >= config.disabled_inherit_rate;
                }
                chosen
            }
        })
        .collect();
    NeatGenome { connections, ..fitter.clone() }
}`,
}

export const distance: Snippet = {
  pseudo: `function compatibility_distance(a, b)
    E ← excess genes, D ← disjoint genes, W ← mean |weight difference| over matching genes
    N ← the larger genome's gene count, or 1 if it has fewer than 20
    return c₁ × E / N + c₂ × D / N + c₃ × W`,
  python: {
    source: SOURCE,
    code: `def compatibility_distance(a, b, config):
    matching, dis_a, dis_b, exc_a, exc_b = align(a, b)
    n = max(len(a.connections), len(b.connections))
    n = 1 if n < 20 else n                          # don't normalize small genomes
    weight_diff = mean(abs(x.weight - y.weight) for x, y in matching)
    return (config.excess_coefficient   * (len(exc_a) + len(exc_b)) / n
          + config.disjoint_coefficient * (len(dis_a) + len(dis_b)) / n
          + config.weight_coefficient   * weight_diff)`,
  },
  rust: `pub fn compatibility_distance(a: &NeatGenome, b: &NeatGenome, config: &NeatConfig) -> f64 {
    let al = align(a, b);
    let n = a.connections.len().max(b.connections.len());
    let n = if n < 20 { 1.0 } else { n as f64 }; // don't normalize small genomes
    let weight_diff = mean(al.matching.iter().map(|(x, y)| (x.weight - y.weight).abs()));
    config.excess_coefficient * (al.excess_a.len() + al.excess_b.len()) as f64 / n
        + config.disjoint_coefficient * (al.disjoint_a.len() + al.disjoint_b.len()) as f64 / n
        + config.weight_coefficient * weight_diff
}`,
}

export const sharing: Snippet = {
  pseudo: `shifted ← each fitness − the lowest fitness + a tiny amount   // Snake fitness can be negative
for each species s
    share(s) ← the MEAN shifted fitness of s's members           // fitness sharing
offspring ← split population_size in proportion to the shares`,
  python: {
    source: SOURCE,
    code: `# fitness can be negative in Snake, so shift it before dividing it up
shifted = [f - min(fitness) + 1e-6 for f in fitness]
# each species' claim on the next generation = its MEAN shifted fitness (fitness sharing)
shares  = [mean(shifted[i] for i in species.members) for species in breeding]
quotas  = allocate(shares, population_size)          # offspring per species, largest remainder`,
  },
  rust: `// fitness can be negative in Snake, so shift it before dividing it up
let lowest = fitness.iter().copied().fold(f64::INFINITY, f64::min);
let shifted: Vec<f64> = fitness.iter().map(|f| f - lowest + 1e-6).collect();
// each species' claim on the next generation = its MEAN shifted fitness (fitness sharing)
let shares: Vec<f64> = breeding.iter().map(|s| mean(s.members.iter().map(|&i| shifted[i]))).collect();
let quotas = allocate(&shares, population_size); // offspring per species, largest remainder`,
}

export const adaptive: Snippet = {
  pseudo: `// Aim for a species COUNT, and let the distance threshold chase it
if there are fewer species than the target then threshold ← threshold − step
if there are more species than the target then threshold ← threshold + step`,
  python: {
    source: SOURCE,
    code: `# A fixed threshold assumes small genomes. Snake starts at 36 genes, where distance is
# divided by N >= 20 -- so structural differences barely register and one species swallows everything.
# Aim for a species COUNT instead, and let the threshold chase it:
if len(species) < config.target_species:
    threshold -= config.threshold_step
elif len(species) > config.target_species:
    threshold += config.threshold_step`,
  },
  rust: `// A fixed threshold assumes small genomes. Snake starts at 36 genes, where distance is
// divided by N >= 20 -- so structural differences barely register and one species swallows everything.
// Aim for a species COUNT instead, and let the threshold chase it:
match species.len().cmp(&config.target_species) {
    Ordering::Less => threshold -= config.threshold_step,
    Ordering::Greater => threshold += config.threshold_step,
    Ordering::Equal => {}
}`,
}
