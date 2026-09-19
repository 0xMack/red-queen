// Two fixed parent genomes for the NEAT chapter's crossover and speciation demos -- the shape of Figure 4 in the
// NEAT paper: parent A has innovations 1-5 and 8, parent B has 1-7 and 9-10, so between them every case
// (matching / disjoint / excess, enabled / disabled) occurs.
//
// Nodes (3 inputs, so: inputs 0-2, bias 3, output 4, hidden 5-6). The innovation numbers name a structural
// change -- the same number means the same (source -> target) connection in every genome, which is the whole
// point of historical markings.
import type { Gene, Genome } from "~/utils/neat"

export const INNOVATION_EDGES: Record<number, [number, number]> = {
  1: [0, 4],
  2: [1, 4],
  3: [2, 4],
  4: [1, 5],
  5: [5, 4],
  6: [5, 6],
  7: [6, 4],
  8: [0, 5],
  9: [2, 6],
  10: [0, 6],
}

const gene = (innovation: number, weight: number, enabled = true): Gene => {
  const [source, target] = INNOVATION_EDGES[innovation]!
  return { innovation, source, target, weight, enabled }
}

export const PARENT_A: Genome = {
  numInputs: 3,
  numOutputs: 1,
  genes: [gene(1, 0.7), gene(2, -0.5, false), gene(3, 0.5), gene(4, 0.2), gene(5, 0.4), gene(8, 0.6)],
}

export const PARENT_B: Genome = {
  numInputs: 3,
  numOutputs: 1,
  genes: [gene(1, 0.6), gene(2, -0.4, false), gene(3, 0.6), gene(4, 0.3), gene(5, 0.2, false), gene(6, 0.5), gene(7, 0.8), gene(9, -0.7), gene(10, 0.9)],
}
