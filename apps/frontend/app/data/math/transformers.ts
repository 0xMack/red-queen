import { add, cases, formula, frac, ln, mul, paren, seq, sqrt, sub, term } from "~/utils/math/expr"

// Transformers' formulas (docs/design/0012): attention and its causal mask (the mask demo's cells are the mask's two
// cases), softmax, a block's two residual steps, layer norm, and the next-character loss -- with a worked row for the
// loss of a model that knows nothing, which is where TinyLM's training started.

const Qm = term("queries", "Q", { name: "the queries", meaning: "One row per position: what that position is looking for." })
const Km = term("keys", "K", { name: "the keys", meaning: "One row per position: what that position contains, to be matched against queries." })
const Vm = term("values", "V", { name: "the values", meaning: "One row per position: what that position passes on to whoever attends to it." })
const dk = term("dk", "d_k", { name: "the head's width", meaning: "16 here. Dividing by √d_k keeps dot products from growing with the width, so the softmax doesn't saturate." })
const mask = term("mask", "M", { name: "the causal mask", meaning: "0 where a position may look, −∞ where it may not: after the softmax, those weights are exactly zero." })

export const attention = formula({
  id: "tf-attention",
  title: "Attention: softmax of the scaled, masked query-key scores, applied to the values",
  lhs: seq("\\operatorname{Attn}(", Qm, ",", Km, ",", Vm, ")"),
  body: mul(
    seq(
      term("softmax", "\\operatorname{softmax}", { name: "softmax", meaning: "Turns each row of scores into weights that are positive and sum to 1." }),
      paren(add(term("scores", frac(seq(Qm, Km, "^{\\top}"), sqrt(dk)), { name: "the scores", meaning: "Every query dotted with every key: how much each position wants each other one." }), mask)),
    ),
    Vm,
  ),
})

export const causalMask = formula({
  id: "tf-mask",
  title: "The causal mask: position i may attend to positions up to i and nothing after",
  lhs: seq(term("mask", "M", { name: "the causal mask" }), "_{ij}"),
  body: cases(
    [term("allowed", "0", { name: "allowed", meaning: "j ≤ i: an earlier (or the same) position -- the score passes unchanged." }), "j \\le i"],
    [term("blocked", "-\\infty", { name: "blocked", meaning: "j > i: a later position -- the answer. In the code, −10⁹, which the softmax turns into a weight of exactly 0." }), "j > i"],
  ),
})

export const softmax = formula({
  id: "tf-softmax",
  title: "Softmax: exponentiate every score and divide by the total",
  lhs: seq("\\operatorname{softmax}(s)_j"),
  body: frac(term("exp-sj", "e^{s_j}", { name: "this score, exponentiated" }), term("sum-exp", "\\sum_k e^{s_k}", { name: "all of the row's, added up" })),
})

const x = term("x", "x", { name: "each position's vector", meaning: "64 numbers per position, carried from block to block." })
const LN = (inner: Parameters<typeof term>[1]) => seq(term("ln", "\\operatorname{LN}", { name: "layer norm" }), paren(inner))

export const blockAttention = formula({
  id: "tf-block-attention",
  title: "The block's first step: add attention's output back onto the input",
  lhs: x,
  rel: "\\leftarrow",
  body: add(x, seq(term("attn", "\\operatorname{Attn}", { name: "attention", meaning: "The positions exchange information." }), paren(LN(x)))),
})

export const blockMlp = formula({
  id: "tf-block-mlp",
  title: "The block's second step: add the MLP's output back onto that",
  lhs: x,
  rel: "\\leftarrow",
  body: add(x, seq(term("mlp", "\\operatorname{MLP}", { name: "the MLP", meaning: "Each position, on its own, processes what it gathered." }), paren(LN(x)))),
})

export const layerNorm = formula({
  id: "tf-layernorm",
  title: "Layer norm: each position's vector rescaled to zero mean and unit variance, then scaled and shifted by learned parameters",
  lhs: LN(x),
  body: add(
    mul(term("ln-scale", "g", { name: "a learned scale" }), frac(sub(x, term("mu", "\\mu", { name: "the vector's mean" })), term("sigma", "\\sigma", { name: "its standard deviation" }))),
    term("ln-shift", "\\beta", { name: "a learned shift" }),
  ),
})

const vocab = term("vocab", "|\\mathcal{V}|", { name: "the vocabulary", meaning: "75 distinct characters in the training text.", num: { decimals: 0, max: 100 } })

export const loss = formula({
  id: "tf-loss",
  title: "The loss: the average negative log-probability the model gave each actual next character",
  lhs: term("loss", "L", { symbol: "loss" }),
  body: seq(
    "-\\frac{1}{T} \\sum_{t=1}^{T} \\log",
    term("p-next", "p_{\\theta}(c_{t+1} \\mid c_{\\le t})", {
      name: "the probability it gave the right next character",
      meaning: "1 would be a perfect prediction (−log 1 = 0); spreading probability over wrong characters is what the loss punishes.",
    }),
  ),
})

// Before training, every character is equally likely: the loss is ln |V|.
export const chanceLoss = formula({
  id: "tf-chance-loss",
  title: "A model that knows nothing gives every character the same probability, so its loss is the log of the vocabulary size",
  lhs: term("loss-0", "L_0", { symbol: "loss", name: "the loss at random initialisation", num: { decimals: 2, max: 9 } }),
  body: seq("-\\log", paren(frac("1", vocab)), "=", ln(vocab)),
  worked: ln(vocab),
  result: "loss-0",
})
