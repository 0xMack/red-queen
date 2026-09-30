import { add, formula, frac, mul, pow, sub, tanh, term } from "~/utils/math/expr"

// Autodiff's formulas (docs/design/0012), for the one-neuron playground: every node of its graph is a term, so the
// formula, the sliders and the graph's circles are one figure. The playground computes the forward and backward
// passes itself; each formula computes them again from the sliders and, in development, checks it gets what the
// playground's backward pass got.

const value = { decimals: 2, max: 9, signed: true }
const x = term("x", "x", { name: "the input", num: value })
const w = term("w", "w", { name: "the weight", meaning: "A parameter: what training changes.", num: value })
const b = term("b", "b", { name: "the bias", meaning: "A parameter: what training changes.", num: value })
const t = term("t", "t", { name: "the target", meaning: "What the neuron should output.", num: value })
const u = term("u", mul(x, w), { name: "u = x · w", num: value })
const z = term("z", add(u, b), { name: "z = u + b", meaning: "The neuron's input, before its activation.", num: value })
const a = term("a", tanh(z), { name: "a = tanh z", meaning: "The neuron's output.", num: value })
const e = term("e", sub(a, t), { name: "e = a − t", meaning: "How far the output is from the target.", num: value })
const L = term("L", "L", { symbol: "loss", num: { decimals: 4, max: 9 } })

export const forward = formula({
  id: "autodiff-forward",
  title: "The playground's loss: the squared difference between the neuron's output and the target",
  lhs: L,
  body: pow(e, "2"),
  worked: pow(e, "2"),
  result: "L",
})

// ∂L/∂w, one local rule per node, multiplied back along the path from L to w -- the chain rule, as backward() runs it.
const partial = (id: string, of: string, by: string, rule: Parameters<typeof term>[1], name: string) =>
  term(id, frac(`\\partial ${of}`, `\\partial ${by}`), { name, expandable: true, forms: { rule: { label: "its local rule", expr: rule } }, num: value })

export const chain = formula({
  id: "autodiff-chain",
  title: "The chain rule: the gradient of the loss with respect to w is the product of each node's local derivative along the path back to w",
  lhs: term("dL-dw", frac("\\partial L", "\\partial w"), { name: "how the loss changes with w", num: { decimals: 4, max: 9, signed: true } }),
  body: mul(
    partial("dL-de", "L", "e", mul("2", e), "L = e²: its derivative is 2e"),
    partial("de-da", "e", "a", "1", "e = a − t: 1"),
    partial("da-dz", "a", "z", sub("1", pow(a, "2")), "a = tanh z: 1 − a²"),
    partial("dz-du", "z", "u", "1", "z = u + b: 1"),
    partial("du-dw", "u", "w", x, "u = x · w: x"),
  ),
  worked: mul(mul("2", e), "1", sub("1", pow(a, "2")), "1", x),
  result: "dL-dw",
  // e and a as the numbers the forward pass computed, not written out again
  atoms: ["e", "a"],
})

export const step = formula({
  id: "autodiff-step",
  title: "A gradient step: move w against its gradient",
  lhs: term("w-new", "w", { name: "the weight after the step", num: value }),
  rel: "\\leftarrow",
  body: sub(w, mul(term("eta", "\\eta", { symbol: "eta", num: { decimals: 2, max: 1 } }), term("dL-dw", frac("\\partial L", "\\partial w"), { name: "how the loss changes with w", num: { decimals: 4, max: 9, signed: true } }))),
  worked: sub(w, mul(term("eta", "\\eta", { symbol: "eta", num: { decimals: 2, max: 1 } }), term("dL-dw", frac("\\partial L", "\\partial w"), { name: "how the loss changes with w", num: { decimals: 4, max: 9, signed: true } }))),
  result: "w-new",
})

export const finiteDifference = formula({
  id: "autodiff-finite-difference",
  title: "The check: the derivative's definition, computed the slow way",
  lhs: term("dL-dw", frac("\\partial L", "\\partial w"), { name: "the gradient backward() computed" }),
  rel: "\\approx",
  body: frac(sub("L(w + \\varepsilon)", "L(w - \\varepsilon)"), "2\\varepsilon"),
})
