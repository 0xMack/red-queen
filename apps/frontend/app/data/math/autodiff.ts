import { add, formula, frac, mul, pow, sqrt, sub, tanh, term } from "~/utils/math/expr"

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

// --- Adam ------------------------------------------------------------------------------------------------------------

const beta1 = term("beta1", "\\beta_1", { name: "the momentum decay", meaning: "0.9: how much of the old running mean of the gradient to keep each step." })
const beta2 = term("beta2", "\\beta_2", { name: "the scale decay", meaning: "0.999: how much of the old running mean of the squared gradient to keep." })
const grad = term("grad", "g_t", { name: "this step's gradient", meaning: "∂L/∂θ for this weight, from backward()." })
const m = (id: string, tex: string, name: string, meaning?: string) => term(id, tex, { name, meaning })

export const adamMomentum = formula({
  id: "adam-m",
  title: "Adam's momentum: a running mean of the gradient",
  lhs: m("m", "m_t", "the momentum", "A running mean of the gradient: noise averages out, a consistent direction builds up."),
  body: add(mul(beta1, m("m-prev", "m_{t-1}", "last step's momentum")), mul(sub("1", beta1), grad)),
})

export const adamScale = formula({
  id: "adam-v",
  title: "Adam's scale: a running mean of the squared gradient",
  lhs: m("v", "v_t", "the scale", "A running mean of the squared gradient: how big this weight's gradients usually are."),
  body: add(mul(beta2, m("v-prev", "v_{t-1}", "last step's scale")), mul(sub("1", beta2), pow(grad, "2"))),
})

export const adamStep = formula({
  id: "adam-step",
  title: "Adam's step: the bias-corrected momentum, divided by the square root of the bias-corrected scale",
  lhs: term("theta", "\\theta", { symbol: "weights" }),
  rel: "\\leftarrow",
  body: sub(
    term("theta", "\\theta", { symbol: "weights" }),
    mul(
      term("eta", "\\eta", { symbol: "eta" }),
      frac(
        term("m-hat", frac(m("m", "m_t", "the momentum"), sub("1", "\\beta_1^{\\,t}")), { name: "the momentum, bias-corrected", meaning: "m starts at 0, so early on it's too small; dividing by 1 − β₁ᵗ undoes that." }),
        add(sqrt(term("v-hat", frac(m("v", "v_t", "the scale"), sub("1", "\\beta_2^{\\,t}")), { name: "the scale, bias-corrected" })), term("adam-eps", "\\epsilon", { name: "a tiny constant", meaning: "10⁻⁸: keeps the division safe for a weight whose gradient has been zero." })),
      ),
    ),
  ),
})
