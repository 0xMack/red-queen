import { add, formula, frac, mul, pow, seq, sub, term } from "~/utils/math/expr"

// Measuring Strength's formulas (docs/design/0012, 0013): Elo's expected score (worked live by the Elo demo from its
// two ratings), what a Bradley-Terry fit makes true, the SPRT's log-likelihood ratio over game pairs (worked live by
// the SPRT lab from its current run) and the test's stopping rule.

// Ratings sit in a small exponent, where fixed-width padding would read as gaps: written plainly (a negative one
// bracketed), at the cost of the row shifting as a slider moves.
const rating = (id: string, tex: string, name: string, meaning: string) =>
  term(id, tex, { name, meaning, format: (v) => (v < 0 ? `(-${Math.abs(Math.round(v))})` : String(Math.round(v))) })
const score = (id: string, tex: string, name: string, meaning: string) => term(id, tex, { name, meaning, num: { decimals: 3, max: 1 } })

const rA = rating("r-a", "R_A", "A's rating", "In Elo points. Only differences mean anything: the leaderboard pins Random at 0.")
const rB = rating("r-b", "R_B", "B's rating", "In Elo points, on the same scale as A's.")
const expectedRhs = frac("1", add("1", pow("10", frac(sub(rB, rA), "400"))))

export const expected = formula({
  id: "strength-expected",
  title: "A's expected points per game against B, from their ratings",
  lhs: score("e-a", "E_A", "A's expected score", "Points per game A is expected to take from B: a win 1, a draw ½. B's is 1 − E_A."),
  body: expectedRhs,
  worked: expectedRhs,
  result: "e-a",
})

export const fitted = formula({
  id: "strength-bradley-terry",
  title: "The fitted ratings: every player's expected points, summed over its games, equal the points it actually scored",
  lhs: seq(
    "\\sum_{j}",
    term("n-ij", "n_{ij}", { name: "games between i and j" }),
    term("e-ij", "E_{ij}", { name: "i's expected score against j", meaning: "From the two ratings, as in the formula above." }),
  ),
  body: term("w-i", "W_i", { name: "the points i actually scored", meaning: "Over every game of the round robin: a win 1, a draw ½." }),
})

const pairs = term("n", "N", { name: "game pairs played", meaning: "One opening, both seats: the independent unit of evidence.", num: { decimals: 0, max: 999 } })
const s0 = score("s0", "s_0", "the score H0 predicts", "The expected score if A is elo0 stronger (0 Elo: 0.500).")
const s1 = score("s1", "s_1", "the score H1 predicts", "The expected score if A is elo1 stronger (50 Elo: 0.571).")
const xbar = score("xbar", "\\bar x", "A's mean pair score", "Points per pair, halved: 0 (lost both), ½ (level), 1 (won both).")
const variance = term("sigma2", "\\sigma^2", { name: "the variance of pair scores", meaning: "How much pairs disagree. Draw-heavy, consistent play has a small one, and every pair counts for more.", num: { decimals: 3, max: 1 } })
const midpoint = score("sbar", "\\bar s", "halfway between the two predictions", "(s₀ + s₁) / 2. A mean pair score above it is evidence for H1, below it for H0.")
// N (s1 - s0)(2 x̄ - s0 - s1) / (2 σ²), written with the midpoint -- the same number, a shorter worked row.
const llrRhs = frac(mul(pairs, sub(s1, s0), sub(xbar, midpoint)), variance)

export const llr = formula({
  id: "strength-llr",
  title: "The log-likelihood ratio: how much more likely the pairs so far are if A is elo1 stronger than if it is elo0 stronger",
  lhs: term("llr", "\\text{LLR}", { name: "the log-likelihood ratio", meaning: "Positive favours H1 (A is stronger), negative H0. It drifts up or down as pairs arrive.", num: { decimals: 2, max: 99, signed: true } }),
  body: llrRhs,
  worked: llrRhs,
  result: "llr",
})

export const stop = formula({
  id: "strength-sprt-stop",
  title: "Keep playing while the LLR is between the two bounds; crossing one decides",
  lhs: seq(term("lower", "\\ln \\tfrac{\\beta}{1-\\alpha}", { name: "the lower bound", meaning: "Cross it and accept H0: A is no stronger. With α = β = 0.05, −2.94." }), "<", term("llr", "\\text{LLR}", { name: "the log-likelihood ratio" })),
  rel: "<",
  body: term("upper", "\\ln \\tfrac{1-\\beta}{\\alpha}", { name: "the upper bound", meaning: "Cross it and accept H1: A is at least elo1 stronger. With α = β = 0.05, +2.94." }),
})
