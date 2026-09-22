// JS mirror of evolve.neuro._forward (tanh on every layer, per-layer [out*in weights][out biases]
// layout) -- used only to *visualize* a champion's activations next to the board. The worker's
// Python copy is still what actually drives the snake; this never decides a move.
export function forwardActivations(weights: number[], layerSizes: number[], observation: number[]): number[][] {
  const layers: number[][] = [observation.slice()]
  let activations = observation
  let offset = 0
  for (let i = 0; i < layerSizes.length - 1; i++) {
    const inSize = layerSizes[i]!
    const outSize = layerSizes[i + 1]!
    const next: number[] = []
    for (let o = 0; o < outSize; o++) {
      let total = weights[offset + inSize * outSize + o]!
      for (let k = 0; k < inSize; k++) total += weights[offset + o * inSize + k]! * activations[k]!
      next.push(Math.tanh(total))
    }
    offset += inSize * outSize + outSize
    activations = next
    layers.push(next)
  }
  return layers
}

/** Weight connecting input k of layer i to output o, same layout as above. */
export function weightAt(weights: readonly number[], layerSizes: readonly number[], layer: number, k: number, o: number): number {
  let offset = 0
  for (let i = 0; i < layer; i++) offset += layerSizes[i]! * layerSizes[i + 1]! + layerSizes[i + 1]!
  return weights[offset + o * layerSizes[layer]! + k]!
}

// games.snake's 11-feature observation, in order (see Snake._observation), and the 3 relative
// actions (argmax index - 1 = -1/0/1), matching the worker's snake_policy_action.
export const SNAKE_INPUT_LABELS = [
  "danger ahead",
  "danger left",
  "danger right",
  "heading →",
  "heading ↓",
  "heading ←",
  "heading ↑",
  "food ←",
  "food →",
  "food ↑",
  "food ↓",
]
// games.snake's `egocentric.v1` (27 values, in order): per ray wall / body / food proximity, then the food and tail
// as (ahead, right) offsets, apples eaten, and the hunger clock. Mirrors `SnakeEgocentric.feature_names`.
const RAY_NAMES = ["left", "front-left", "front", "front-right", "right", "back-left", "back-right"]
export const SNAKE_EGOCENTRIC_LABELS = [
  ...RAY_NAMES.flatMap((ray) => ["wall", "body", "food"].map((kind) => `${ray} ${kind}`)),
  "food ahead",
  "food right",
  "tail ahead",
  "tail right",
  "apples eaten",
  "hunger",
]
export const SNAKE_OUTPUT_LABELS = ["turn left", "straight", "turn right"]

/** The input labels for a Snake network of `numInputs` inputs (the observers differ in size), or none if unknown. */
export function snakeInputLabels(numInputs: number): string[] {
  if (numInputs === SNAKE_INPUT_LABELS.length) return SNAKE_INPUT_LABELS
  if (numInputs === SNAKE_EGOCENTRIC_LABELS.length) return SNAKE_EGOCENTRIC_LABELS
  return []
}
