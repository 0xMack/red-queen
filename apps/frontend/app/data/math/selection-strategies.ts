import { formula, frac, seq, sub, term } from "~/utils/math/expr"

// Selection Strategies' formulas (docs/design/0012): the mean fitness tournament selection ranks by, the tournament
// itself, lexicase's per-case filter (worked on an example case), its tolerance, and Pareto dominance.

const meanF = term("mean-fitness", "\\bar F(g)", { symbol: "fitness", name: "the genome's mean fitness", meaning: "Its fitness averaged over every test case: what tournament selection compares." })
const fc = term("case-fitness", "f_c(g)", { name: "its fitness on case c", meaning: "One number per test case (per game, per data point) -- lexicase never averages them." })

export const mean = formula({
  id: "sel-mean",
  title: "Mean fitness: a genome's fitness averaged over every test case",
  lhs: meanF,
  body: seq(frac("1", term("cases", "|C|", { name: "the number of test cases" })), "\\sum_{c \\in C}", fc),
})

export const tournament = formula({
  id: "sel-tournament",
  title: "Tournament selection: the best mean fitness among k genomes drawn at random",
  lhs: seq("\\text{parent}"),
  body: seq(
    "\\operatorname*{arg\\,max}_{g \\in T}",
    meanF,
    ",\\qquad",
    term("tournament", "T", { name: "the tournament", meaning: "k genomes drawn uniformly at random, with replacement." }),
    "= k \\text{ random draws}",
  ),
})

const best = term("best", "\\max_{h} f_c(h)", { name: "the best score on this case", meaning: "Among the candidates still in the running.", num: { decimals: 2, max: 9 } })
const eps = term("epsilon-c", "\\varepsilon_c", {
  name: "the case's tolerance",
  meaning: "The median absolute deviation of the candidates' scores on this case: how far from the best still counts as 'as good', for a continuous fitness where exact ties are rare.",
  num: { decimals: 2, max: 9 },
})

export const lexicaseCut = formula({
  id: "sel-lexicase-cut",
  title: "Lexicase, one case at a time: keep the candidates within epsilon of the best on this case",
  lhs: seq("\\text{keep } g \\text{ if }", fc),
  rel: "\\ge",
  body: term("cut", sub(best, eps), { name: "the cut-off on this case", num: { decimals: 2, max: 9 } }),
  worked: sub(best, eps),
  result: "cut",
  workedLhs: seq("\\text{cut-off}"),
})

export const lexicaseEpsilon = formula({
  id: "sel-lexicase-epsilon",
  title: "Lexicase's tolerance: the median of each candidate's distance from the median score",
  lhs: eps,
  body: seq("\\operatorname{median}_i \\left|", term("fi", "f_c(i)", { name: "candidate i's score on this case" }), "-", "\\operatorname{median}_j f_c(j) \\right|"),
})

export const dominance = formula({
  id: "sel-pareto",
  title: "Pareto dominance: at least as fit and at least as simple, and strictly better on one of the two",
  lhs: seq("A \\succ B"),
  rel: "\\iff",
  body: seq(
    term("fitter", "\\bar F_A \\ge \\bar F_B", { name: "at least as fit" }),
    "\\;\\wedge\\;",
    term("simpler", "c_A \\le c_B", { name: "at least as simple", meaning: "c is the injected complexity measure: for linear GP, the instructions that can reach the output." }),
    "\\;\\wedge\\;",
    term("strictly", "(\\bar F_A > \\bar F_B \\,\\vee\\, c_A < c_B)", { name: "and strictly better on one" }),
  ),
})
