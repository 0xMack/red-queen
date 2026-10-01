import { add, formula, frac, pow, seq, sub, term } from "~/utils/math/expr"

// Who to Play's formulas (docs/design/0012, 0014): prioritized fictitious self-play's weight for a past self, worked by
// the PFSP demo for the member being hovered, and its running score.

const score = term("s-i", "s_i", {
  name: "the network's running score against this past self",
  meaning: "Win 1, draw ½, averaged over its recent games against it (an exponential average). It starts at ½.",
  num: { decimals: 2, max: 1 },
})
const p = term("p", "p", {
  name: "how sharply to prioritize",
  meaning: "0 plays every past self equally often; 2 (the setting used here) sends most games to the ones it still loses to.",
  num: { decimals: 1, max: 9 },
})
const weightRhs = add(pow(sub("1", score), p), "0.01")

export const weight = formula({
  id: "regimes-pfsp-weight",
  title: "A past self's weight: large when the network still fails to beat it, never quite zero",
  lhs: term("w-i", "w_i", { name: "this past self's weight", num: { decimals: 3, max: 9 } }),
  body: weightRhs,
  worked: weightRhs,
  result: "w-i",
})

export const choice = formula({
  id: "regimes-pfsp-choice",
  title: "The chance a pool game is against this past self: its share of the weights",
  lhs: term("pick-i", "P(i)", { name: "the chance of drawing past self i" }),
  body: frac(term("w-i", "w_i", { name: "this past self's weight" }), seq("\\sum_j", term("w-j", "w_j", { name: "every past self's weight" }))),
})

export const running = formula({
  id: "regimes-running-score",
  title: "After each game against past self i, its score moves 5% of the way toward the result",
  lhs: term("s-i", "s_i", { name: "the running score" }),
  rel: "\\leftarrow",
  body: add(term("s-i", "s_i", { name: "the running score" }), seq("0.05", "\\,", "(", term("result", "r", { name: "this game's result", meaning: "1 for a win, ½ for a draw, 0 for a loss." }), "-", term("s-i", "s_i", { name: "the running score" }), ")")),
})
