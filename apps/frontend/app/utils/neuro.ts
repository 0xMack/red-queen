// The fixed-topology counterpart to utils/neat.ts, for the neuroevolution chapter's demos: a genome is one
// flat list of numbers (evolve.neuro.WeightVector's layout: per layer, `out*in` weights then `out` biases),
// and evolution is Evolution Strategies -- keep the best few, add Gaussian noise. Same rules as
// libs/evolve/src/evolve/neuro.py (GaussianMutation, tanh on every layer), minus lexicase: the demos use
// plain truncation selection, which is enough to show the mechanism.
import { forwardActivations } from "~/utils/snakePolicy"
import { Rng, xorFitness } from "~/utils/neat"

export const XOR_SHAPE = [2, 3, 1]

export const paramCount = (layers: number[]) => layers.slice(1).reduce((n, size, i) => n + size * (layers[i]! + 1), 0)

export const randomWeights = (layers: number[], rng: Rng, scale = 1) => Array.from({ length: paramCount(layers) }, () => rng.uniform(-scale, scale))

export const output = (weights: number[], layers: number[], input: number[]) => forwardActivations(weights, layers, input).at(-1)![0]!

export const xorWeightsFitness = (weights: number[], layers = XOR_SHAPE) => xorFitness((x) => output(weights, layers, x))

/** What flat index `i` means: which layer, weight or bias, and which units it connects. */
export function describeIndex(layers: number[], i: number): { layer: number; kind: "weight" | "bias"; from?: number; to: number } {
  let offset = 0
  for (let l = 0; l < layers.length - 1; l++) {
    const inSize = layers[l]!
    const outSize = layers[l + 1]!
    const weightsEnd = offset + inSize * outSize
    if (i < weightsEnd) return { layer: l, kind: "weight", from: (i - offset) % inSize, to: Math.floor((i - offset) / inSize) }
    if (i < weightsEnd + outSize) return { layer: l, kind: "bias", to: i - weightsEnd }
    offset = weightsEnd + outSize
  }
  throw new RangeError(`index ${i} is outside a ${layers.join("-")} network`)
}

/** Index ranges of each segment of the flat vector, in order: [{layer, kind, start, end}]. */
export function segments(layers: number[]): { layer: number; kind: "weights" | "biases"; start: number; end: number }[] {
  const out: { layer: number; kind: "weights" | "biases"; start: number; end: number }[] = []
  let offset = 0
  for (let l = 0; l < layers.length - 1; l++) {
    const n = layers[l]! * layers[l + 1]!
    out.push({ layer: l, kind: "weights", start: offset, end: offset + n })
    out.push({ layer: l, kind: "biases", start: offset + n, end: offset + n + layers[l + 1]! })
    offset += n + layers[l + 1]!
  }
  return out
}

export interface EsReport {
  generation: number
  best: number
  mean: number
  fitnesses: number[]
  champion: number[]
}

/** Evolution Strategies on a flat weight vector, one generation per step(). */
export class EsRun {
  population: number[][]
  generation = 0
  readonly rng: Rng
  constructor(
    readonly layers: number[],
    readonly size: number,
    public sigma: number,
    readonly evaluate: (weights: number[]) => number,
    seed: number,
    readonly survivors = 0.2,
  ) {
    this.rng = new Rng(seed)
    this.population = Array.from({ length: size }, () => randomWeights(layers, this.rng))
  }

  step(): EsReport {
    // arrow, not `map(this.evaluate)`: map would pass the index as a second argument, and evaluators have optional params
    const fitnesses = this.population.map((w) => this.evaluate(w))
    const ranked = fitnesses.map((_, i) => i).sort((a, b) => fitnesses[b]! - fitnesses[a]!)
    const report: EsReport = {
      generation: this.generation,
      best: fitnesses[ranked[0]!]!,
      mean: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
      fitnesses,
      champion: this.population[ranked[0]!]!,
    }
    const parents = ranked.slice(0, Math.max(1, Math.ceil(this.size * this.survivors)))
    const next = [this.population[ranked[0]!]!] // elitism: the champion survives unchanged
    while (next.length < this.size) {
      const parent = this.population[this.rng.choice(parents)]!
      next.push(parent.map((w) => w + this.rng.gauss(0, this.sigma)))
    }
    this.population = next
    this.generation++
    return report
  }
}
