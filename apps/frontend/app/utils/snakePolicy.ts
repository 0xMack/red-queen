// JS forward pass of a layered network (per layer, [out*in weights][out biases]; evolve.neuro's tanh
// throughout unless per-layer activations say otherwise) -- used only to *visualize* a champion's
// activations next to the board. The model package is what actually plays; this never decides a move.
export function forwardActivations(
  weights: number[],
  layerSizes: number[],
  observation: number[],
  // one per layer of weights; evolved networks are tanh throughout, an RL Q-network (docs/design/0010) is ReLU then linear
  layerActivations: readonly string[] | null = null,
): number[][] {
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
      const activation = layerActivations?.[i] ?? "tanh"
      next.push(activation === "relu" ? Math.max(total, 0) : activation === "linear" ? total : Math.tanh(total))
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
// `egocentric.v2` (33): egocentric.v1, then per move the share of free cells still reachable and whether the tail is.
// Mirrors `SnakeEgocentricV2.feature_names`.
export const SNAKE_EGOCENTRIC_V2_LABELS = [
  ...SNAKE_EGOCENTRIC_LABELS,
  ...["left", "straight", "right"].map((m) => `space ${m}`),
  ...["left", "straight", "right"].map((m) => `tail reachable ${m}`),
]
export const SNAKE_OUTPUT_LABELS = ["turn left", "straight", "turn right"]

/** The input labels for a Snake network of `numInputs` inputs (the observers differ in size), or none if unknown. */
export function snakeInputLabels(numInputs: number): string[] {
  if (numInputs === SNAKE_INPUT_LABELS.length) return SNAKE_INPUT_LABELS
  if (numInputs === SNAKE_EGOCENTRIC_LABELS.length) return SNAKE_EGOCENTRIC_LABELS
  if (numInputs === SNAKE_EGOCENTRIC_V2_LABELS.length) return SNAKE_EGOCENTRIC_V2_LABELS
  return []
}
