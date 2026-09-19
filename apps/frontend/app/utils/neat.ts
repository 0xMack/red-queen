// A TypeScript port of libs/evolve/src/evolve/neat.py -- the same genome, the same mutations, the same
// innovation-aligned crossover, compatibility distance, speciation, and reproduction -- so the Learn
// chapters' demos can evolve real NEAT networks instantly in the browser instead of waiting on Pyodide's
// ~10s cold load. Keep the two in sync by hand (like app/types/*.ts): forward()/activations() are
// checked against the Python on the same genome JSON, and the algorithm's constants mirror NeatConfig.
//
// Also used to *draw* trained Snake champions (the Python genome's JSON is exactly `Genome` below).

export interface Gene {
  innovation: number
  source: number
  target: number
  weight: number
  enabled: boolean
}

export interface Genome {
  numInputs: number
  numOutputs: number
  genes: Gene[] // sorted by innovation
}

/** The wire format `NeatGenome.to_json()` writes (snake_case), as stored in run artifacts. */
export interface GenomeJson {
  type: "neat"
  num_inputs: number
  num_outputs: number
  connections: Gene[]
}

export function genomeFromJson(json: GenomeJson): Genome {
  return { numInputs: json.num_inputs, numOutputs: json.num_outputs, genes: json.connections.map((g) => ({ ...g })) }
}

// Node ids by role (same layout as NeatGenome): inputs, bias, outputs, then hidden.
export const biasId = (g: Genome) => g.numInputs
export const firstOutputId = (g: Genome) => g.numInputs + 1
export const firstHiddenId = (g: Genome) => g.numInputs + 1 + g.numOutputs
export const outputIds = (g: Genome) => Array.from({ length: g.numOutputs }, (_, i) => firstOutputId(g) + i)

export function hiddenIds(g: Genome): number[] {
  const first = firstHiddenId(g)
  const ids = new Set<number>()
  for (const c of g.genes) {
    if (c.source >= first) ids.add(c.source)
    if (c.target >= first) ids.add(c.target)
  }
  return [...ids].sort((a, b) => a - b)
}

export function complexity(g: Genome): { hidden: number; connections: number } {
  return { hidden: hiddenIds(g).length, connections: g.genes.filter((c) => c.enabled).length }
}

// --- Forward pass ------------------------------------------------------------------------------------

/** Every live node's activation (node id -> value): tanh on hidden and output nodes, bias fixed at 1. */
export function activations(g: Genome, observation: number[]): Map<number, number> {
  const incoming = new Map<number, Gene[]>()
  for (const c of g.genes) if (c.enabled) (incoming.get(c.target) ?? incoming.set(c.target, []).get(c.target)!).push(c)

  const outs = outputIds(g)
  const first = firstHiddenId(g)
  const needed = new Set<number>(outs)
  const stack = [...outs]
  while (stack.length) {
    for (const c of incoming.get(stack.pop()!) ?? []) {
      if (c.source >= first && !needed.has(c.source)) {
        needed.add(c.source)
        stack.push(c.source)
      }
    }
  }

  const values = new Map<number, number>()
  observation.forEach((v, i) => values.set(i, v))
  values.set(biasId(g), 1)

  // Kahn's algorithm over the needed hidden/output nodes; inputs/bias are always ready.
  const remaining = new Map<number, number>()
  const dependents = new Map<number, number[]>()
  for (const n of needed) {
    let k = 0
    for (const c of incoming.get(n) ?? []) {
      if (needed.has(c.source)) {
        k++
        ;(dependents.get(c.source) ?? dependents.set(c.source, []).get(c.source)!).push(n)
      }
    }
    remaining.set(n, k)
  }
  const ready = [...needed].filter((n) => remaining.get(n) === 0).sort((a, b) => a - b)
  while (ready.length) {
    const n = ready.pop()!
    let total = 0
    for (const c of incoming.get(n) ?? []) total += c.weight * (values.get(c.source) ?? 0)
    values.set(n, Math.tanh(total))
    for (const d of dependents.get(n) ?? []) {
      remaining.set(d, remaining.get(d)! - 1)
      if (remaining.get(d) === 0) ready.push(d)
    }
  }
  return values
}

export function forward(g: Genome, observation: number[]): number[] {
  const values = activations(g, observation)
  return outputIds(g).map((id) => values.get(id) ?? 0)
}

// --- Randomness --------------------------------------------------------------------------------------

/** Small seeded PRNG (mulberry32) so a demo run is reproducible from its seed. */
export class Rng {
  private state: number
  constructor(seed: number) {
    this.state = seed >>> 0
  }
  random(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0
    let t = this.state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  uniform(a: number, b: number): number {
    return a + (b - a) * this.random()
  }
  gauss(mean = 0, sd = 1): number {
    const u = 1 - this.random()
    const v = this.random()
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  int(n: number): number {
    return Math.floor(this.random() * n)
  }
  choice<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!
  }
}

// --- Historical markings -----------------------------------------------------------------------------

export class InnovationTracker {
  private connections = new Map<string, number>()
  private splits = new Map<number, number>()
  private nextInnovation = 0
  private nextNode: number
  constructor(firstHiddenId: number) {
    this.nextNode = firstHiddenId
  }
  connection(source: number, target: number): number {
    const key = `${source}>${target}`
    if (!this.connections.has(key)) this.connections.set(key, this.nextInnovation++)
    return this.connections.get(key)!
  }
  splitNode(innovation: number): number {
    if (!this.splits.has(innovation)) this.splits.set(innovation, this.nextNode++)
    return this.splits.get(innovation)!
  }
  knownSplitNode(innovation: number): number | undefined {
    return this.splits.get(innovation)
  }
  get innovations(): number {
    return this.nextInnovation
  }
}

export function initialGenome(numInputs: number, numOutputs: number, tracker: InnovationTracker, rng: Rng, scale = 1): Genome {
  const genes: Gene[] = []
  for (let o = 0; o < numOutputs; o++) {
    for (let s = 0; s <= numInputs; s++) {
      const target = numInputs + 1 + o
      genes.push({ innovation: tracker.connection(s, target), source: s, target, weight: rng.uniform(-scale, scale), enabled: true })
    }
  }
  return { numInputs, numOutputs, genes: genes.sort((a, b) => a.innovation - b.innovation) }
}

// --- Configuration -----------------------------------------------------------------------------------

export interface NeatConfig {
  weightMutationRate: number
  weightPerturbSigma: number
  weightReplaceRate: number
  weightReplaceScale: number
  addConnectionRate: number
  addNodeRate: number
  toggleRate: number
  crossoverRate: number
  interspeciesRate: number
  disabledInheritRate: number
  compatibilityThreshold: number
  excessCoefficient: number
  disjointCoefficient: number
  weightCoefficient: number
  survivalThreshold: number
  stagnationLimit: number
  minSpeciesKept: number
  speciesElitismSize: number
  maxHiddenNodes: number
  speciation: boolean
}

// Same defaults as evolve.neat.NeatConfig.
export const DEFAULT_CONFIG: NeatConfig = {
  weightMutationRate: 0.8,
  weightPerturbSigma: 0.2,
  weightReplaceRate: 0.1,
  weightReplaceScale: 1,
  addConnectionRate: 0.08,
  addNodeRate: 0.04,
  toggleRate: 0.01,
  crossoverRate: 0.75,
  interspeciesRate: 0.001,
  disabledInheritRate: 0.75,
  compatibilityThreshold: 3,
  excessCoefficient: 1,
  disjointCoefficient: 1,
  weightCoefficient: 0.4,
  survivalThreshold: 0.2,
  stagnationLimit: 15,
  minSpeciesKept: 2,
  speciesElitismSize: 5,
  maxHiddenNodes: 40,
  speciation: true,
}

// --- Variation ---------------------------------------------------------------------------------------

/** Would adding source -> target close a loop? True iff `source` is already reachable from `target`. */
export function createsCycle(g: Genome, source: number, target: number): boolean {
  if (source === target) return true
  const successors = new Map<number, number[]>()
  for (const c of g.genes) (successors.get(c.source) ?? successors.set(c.source, []).get(c.source)!).push(c.target)
  const stack = [target]
  const seen = new Set([target])
  while (stack.length) {
    const n = stack.pop()!
    if (n === source) return true
    for (const next of successors.get(n) ?? []) {
      if (!seen.has(next)) {
        seen.add(next)
        stack.push(next)
      }
    }
  }
  return false
}

const bySorted = (genes: Gene[]) => genes.sort((a, b) => a.innovation - b.innovation)

export function mutateWeights(g: Genome, cfg: NeatConfig, rng: Rng): Genome {
  return {
    ...g,
    genes: g.genes.map((c) => ({
      ...c,
      weight: rng.random() < cfg.weightReplaceRate ? rng.uniform(-cfg.weightReplaceScale, cfg.weightReplaceScale) : c.weight + rng.gauss(0, cfg.weightPerturbSigma),
    })),
  }
}

/** The (source, target) pairs add_connection could still pick -- for showing what's legal. */
export function legalNewConnections(g: Genome): [number, number][] {
  const sources = [...Array.from({ length: firstOutputId(g) }, (_, i) => i), ...hiddenIds(g)]
  const targets = [...outputIds(g), ...hiddenIds(g)]
  const existing = new Set(g.genes.map((c) => `${c.source}>${c.target}`))
  const legal: [number, number][] = []
  for (const s of sources) for (const t of targets) if (!existing.has(`${s}>${t}`) && !createsCycle(g, s, t)) legal.push([s, t])
  return legal
}

export function addConnection(g: Genome, tracker: InnovationTracker, cfg: NeatConfig, rng: Rng): Genome {
  const legal = legalNewConnections(g)
  if (legal.length === 0) return g
  const [source, target] = rng.choice(legal)
  const gene: Gene = { innovation: tracker.connection(source, target), source, target, weight: rng.uniform(-cfg.weightReplaceScale, cfg.weightReplaceScale), enabled: true }
  return { ...g, genes: bySorted([...g.genes, gene]) }
}

export function addNode(g: Genome, tracker: InnovationTracker, cfg: NeatConfig, rng: Rng): Genome {
  if (hiddenIds(g).length >= cfg.maxHiddenNodes) return g
  const present = new Set(hiddenIds(g))
  const candidates = g.genes.filter((c) => {
    if (!c.enabled || c.source === biasId(g)) return false
    const known = tracker.knownSplitNode(c.innovation)
    return known === undefined || !present.has(known)
  })
  if (candidates.length === 0) return g
  const old = rng.choice(candidates)
  const node = tracker.splitNode(old.innovation)
  const into: Gene = { innovation: tracker.connection(old.source, node), source: old.source, target: node, weight: 1, enabled: true }
  const out: Gene = { innovation: tracker.connection(node, old.target), source: node, target: old.target, weight: old.weight, enabled: true }
  const kept = g.genes.map((c) => (c.innovation === old.innovation ? { ...c, enabled: false } : c))
  return { ...g, genes: bySorted([...kept, into, out]) }
}

export function toggleConnection(g: Genome, rng: Rng): Genome {
  const i = rng.int(g.genes.length)
  return { ...g, genes: g.genes.map((c, k) => (k === i ? { ...c, enabled: !c.enabled } : c)) }
}

export function mutate(g: Genome, cfg: NeatConfig, tracker: InnovationTracker, rng: Rng): Genome {
  let out = g
  if (rng.random() < cfg.weightMutationRate) out = mutateWeights(out, cfg, rng)
  if (rng.random() < cfg.addConnectionRate) out = addConnection(out, tracker, cfg, rng)
  if (rng.random() < cfg.addNodeRate) out = addNode(out, tracker, cfg, rng)
  if (rng.random() < cfg.toggleRate) out = toggleConnection(out, rng)
  return out
}

export interface Alignment {
  matching: [Gene, Gene][]
  disjointA: Gene[]
  disjointB: Gene[]
  excessA: Gene[]
  excessB: Gene[]
}

/** Line two genomes up by innovation number (see evolve.neat.align). */
export function align(a: Genome, b: Genome): Alignment {
  const byA = new Map(a.genes.map((c) => [c.innovation, c]))
  const byB = new Map(b.genes.map((c) => [c.innovation, c]))
  const maxA = Math.max(-1, ...byA.keys())
  const maxB = Math.max(-1, ...byB.keys())
  const matching: [Gene, Gene][] = [...byA.keys()].filter((i) => byB.has(i)).sort((x, y) => x - y).map((i) => [byA.get(i)!, byB.get(i)!])
  const onlyA = [...byA.keys()].filter((i) => !byB.has(i)).sort((x, y) => x - y).map((i) => byA.get(i)!)
  const onlyB = [...byB.keys()].filter((i) => !byA.has(i)).sort((x, y) => x - y).map((i) => byB.get(i)!)
  return {
    matching,
    disjointA: onlyA.filter((c) => c.innovation <= maxB),
    disjointB: onlyB.filter((c) => c.innovation <= maxA),
    excessA: onlyA.filter((c) => c.innovation > maxB),
    excessB: onlyB.filter((c) => c.innovation > maxA),
  }
}

/** Offspring of two parents (`fitter` first). Matching genes: either parent's, at random; the rest: fitter's. */
export function crossover(fitter: Genome, other: Genome, cfg: NeatConfig, rng: Rng): Genome {
  const otherBy = new Map(other.genes.map((c) => [c.innovation, c]))
  const genes = fitter.genes.map((f) => {
    const o = otherBy.get(f.innovation)
    if (!o) return { ...f }
    const chosen = rng.random() < 0.5 ? f : o
    let enabled = chosen.enabled
    if (!(f.enabled && o.enabled)) enabled = rng.random() >= cfg.disabledInheritRate
    return { ...chosen, enabled }
  })
  return { ...fitter, genes }
}

export interface DistanceParts {
  excess: number
  disjoint: number
  meanWeightDiff: number
  n: number
  distance: number
}

/** delta = c1*E/N + c2*D/N + c3*W -- with the parts, so a demo can show where the number comes from. */
export function compatibility(a: Genome, b: Genome, cfg: Pick<NeatConfig, "excessCoefficient" | "disjointCoefficient" | "weightCoefficient">): DistanceParts {
  const { matching, disjointA, disjointB, excessA, excessB } = align(a, b)
  const size = Math.max(a.genes.length, b.genes.length)
  const n = size < 20 ? 1 : size
  const meanWeightDiff = matching.length ? matching.reduce((s, [x, y]) => s + Math.abs(x.weight - y.weight), 0) / matching.length : 0
  const excess = excessA.length + excessB.length
  const disjoint = disjointA.length + disjointB.length
  return {
    excess,
    disjoint,
    meanWeightDiff,
    n,
    distance: (cfg.excessCoefficient * excess) / n + (cfg.disjointCoefficient * disjoint) / n + cfg.weightCoefficient * meanWeightDiff,
  }
}

// --- The loop, one generation at a time --------------------------------------------------------------

export interface SpeciesState {
  id: number
  representative: Genome
  members: number[]
  bestFitness: number
  lastImproved: number
}

export interface GenerationReport {
  generation: number
  best: number
  mean: number
  champion: Genome
  fitnesses: number[]
  speciesSizes: { id: number; size: number }[]
}

function allocate(shares: number[], total: number): number[] {
  const weight = shares.reduce((a, b) => a + b, 0)
  let base: number[]
  if (weight <= 0) {
    base = shares.map(() => Math.floor(total / shares.length))
  } else {
    const exact = shares.map((s) => (total * s) / weight)
    base = exact.map(Math.floor)
    const order = exact.map((_, i) => i).sort((i, j) => exact[j]! - base[j]! - (exact[i]! - base[i]!))
    let left = total - base.reduce((a, b) => a + b, 0)
    for (const i of order) {
      if (left-- <= 0) break
      base[i]!++
    }
  }
  let left = total - base.reduce((a, b) => a + b, 0)
  for (let i = 0; left > 0; i++, left--) base[i % base.length]!++
  return base
}

/**
 * NEAT on a small problem, stepped a generation at a time so a page can animate it. `evaluate` scores one
 * genome (higher = better). Mirrors evolve.neat.evolve_neat: evaluate -> speciate -> (report) -> reproduce.
 */
export class NeatRun {
  population: Genome[]
  species: SpeciesState[] = []
  generation = 0
  readonly tracker: InnovationTracker
  readonly rng: Rng
  private nextSpeciesId = 0

  readonly numInputs: number
  readonly numOutputs: number
  readonly cfg: NeatConfig
  readonly evaluate: (g: Genome) => number

  constructor(numInputs: number, numOutputs: number, populationSize: number, cfg: NeatConfig, evaluate: (g: Genome) => number, seed: number) {
    this.numInputs = numInputs
    this.numOutputs = numOutputs
    this.cfg = cfg
    this.evaluate = evaluate
    this.rng = new Rng(seed)
    this.tracker = new InnovationTracker(numInputs + 1 + numOutputs)
    this.population = Array.from({ length: populationSize }, () => initialGenome(numInputs, numOutputs, this.tracker, this.rng))
  }

  private speciate() {
    for (const s of this.species) s.members = []
    if (!this.cfg.speciation) {
      if (this.species.length === 0) this.species = [{ id: this.nextSpeciesId++, representative: this.population[0]!, members: [], bestFitness: -Infinity, lastImproved: this.generation }]
      this.species = this.species.slice(0, 1)
      this.species[0]!.members = this.population.map((_, i) => i)
      return
    }
    this.population.forEach((g, index) => {
      const home = this.species.find((s) => compatibility(g, s.representative, this.cfg).distance < this.cfg.compatibilityThreshold)
      if (home) home.members.push(index)
      else this.species.push({ id: this.nextSpeciesId++, representative: g, members: [index], bestFitness: -Infinity, lastImproved: this.generation })
    })
    this.species = this.species.filter((s) => s.members.length > 0)
  }

  step(): GenerationReport {
    const fitnesses = this.population.map((g) => this.evaluate(g)) // not map(this.evaluate): map's index argument would leak in
    const ranked = fitnesses.map((_, i) => i).sort((a, b) => fitnesses[b]! - fitnesses[a]!)
    this.speciate()
    for (const s of this.species) {
      const best = Math.max(...s.members.map((i) => fitnesses[i]!))
      if (best > s.bestFitness) {
        s.bestFitness = best
        s.lastImproved = this.generation
      }
    }
    const report: GenerationReport = {
      generation: this.generation,
      best: fitnesses[ranked[0]!]!,
      mean: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
      champion: this.population[ranked[0]!]!,
      fitnesses,
      speciesSizes: this.species.map((s) => ({ id: s.id, size: s.members.length })),
    }
    for (const s of this.species) s.representative = this.population[this.rng.choice(s.members)]!
    this.population = this.reproduce(fitnesses)
    this.generation++
    return report
  }

  private reproduce(fitnesses: number[]): Genome[] {
    const { cfg, rng, population } = this
    const size = population.length
    const floor = Math.min(...fitnesses)
    const shifted = fitnesses.map((f) => f - floor + 1e-6)
    const byBest = [...this.species].sort((a, b) => b.bestFitness - a.bestFitness)
    const protectedIds = new Set(byBest.slice(0, cfg.minSpeciesKept).map((s) => s.id))
    const breeding = this.species.filter((s) => this.generation - s.lastImproved < cfg.stagnationLimit || protectedIds.has(s.id))
    const shares = breeding.map((s) => s.members.reduce((a, i) => a + shifted[i]!, 0) / s.members.length)
    const quotas = allocate(shares, size)
    const champion = fitnesses.indexOf(Math.max(...fitnesses))

    const next: Genome[] = []
    breeding.forEach((s, k) => {
      let quota = quotas[k]!
      if (quota === 0) return
      const members = [...s.members].sort((a, b) => fitnesses[b]! - fitnesses[a]!)
      if (members.length >= cfg.speciesElitismSize || members.includes(champion)) {
        next.push(population[members[0]!]!)
        quota--
      }
      const parents = members.slice(0, Math.max(1, Math.ceil(members.length * cfg.survivalThreshold)))
      for (let q = 0; q < quota; q++) {
        const a = rng.choice(parents)
        let child = population[a]!
        if (parents.length > 1 && rng.random() < cfg.crossoverRate) {
          let b = rng.choice(parents)
          if (rng.random() < cfg.interspeciesRate && breeding.length > 1) b = rng.choice(rng.choice(breeding.filter((x) => x !== s)).members)
          const [fit, weak] = fitnesses[a]! >= fitnesses[b]! ? [a, b] : [b, a]
          child = crossover(population[fit]!, population[weak]!, cfg, rng)
        }
        next.push(mutate(child, cfg, this.tracker, rng))
      }
    })
    const ranked = fitnesses.map((_, i) => i).sort((a, b) => fitnesses[b]! - fitnesses[a]!)
    while (next.length < size) next.push(mutate(population[rng.choice(ranked.slice(0, Math.max(1, Math.floor(size / 5))))]!, cfg, this.tracker, rng))
    return next.slice(0, size)
  }
}

// --- XOR, the demos' benchmark -----------------------------------------------------------------------

export const XOR_CASES: { input: number[]; target: number }[] = [
  { input: [0, 0], target: 0 },
  { input: [0, 1], target: 1 },
  { input: [1, 0], target: 1 },
  { input: [1, 1], target: 0 },
]

/** Mean of 1 - squared error over the four rows, outputs mapped from tanh's [-1, 1] to [0, 1]. A network
 *  with no hidden structure can reach at most 0.75 (it can't separate XOR); 1.0 is perfect. */
export function xorFitness(output: (input: number[]) => number): number {
  return XOR_CASES.reduce((s, { input, target }) => s + 1 - ((output(input) + 1) / 2 - target) ** 2, 0) / XOR_CASES.length
}

export const xorGenomeFitness = (g: Genome) => xorFitness((x) => forward(g, x)[0]!)
