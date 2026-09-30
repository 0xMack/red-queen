import { add, formula, frac, mul, pow, seq, sub, term } from "~/utils/math/expr"

// Genetic Algorithms' formulas (docs/design/0012): the curve the baseline run evolves toward (worked at one of its 11
// sample points), the per-case fitness SymbolicRegressionFitness computes, and the mean the loop ranks genomes by.

const x = term("x", "x", { name: "a sample point", meaning: "One of 11 inputs, −1 to 1 in steps of 0.2.", num: { decimals: 1, max: 1, signed: true } })

export const target = formula({
  id: "ga-target",
  title: "The target curve the run evolves a program to match",
  lhs: term("y", "y(x)", { name: "the target", num: { decimals: 4, max: 9 } }),
  body: add(sub(pow(x, "4"), mul("3", pow(x, "2"))), "2"),
  worked: add(sub(pow(x, "4"), mul("3", pow(x, "2"))), "2"),
  result: "y",
})

const g = term("g-x", "g(x_c)", { name: "the program's output on case c" })
const yc = term("y-c", "y(x_c)", { name: "the target's value on case c" })

export const caseFitness = formula({
  id: "ga-case-fitness",
  title: "Fitness on one test case: the negative squared error of the program's output",
  lhs: term("case-fitness", "f_c(g)", { name: "the program's fitness on case c", meaning: "0 is perfect; every miss makes it more negative -- so higher is better, as everywhere in this project." }),
  body: seq("-", pow(sub(g, yc), "2")),
})

export const meanFitness = formula({
  id: "ga-mean-fitness",
  title: "A program's fitness: the mean of its per-case fitnesses over the 11 sample points",
  lhs: term("fitness", "F(g)", { symbol: "fitness", name: "the program's fitness" }),
  body: seq(frac("1", "11"), "\\sum_{c=1}^{11}", term("case-fitness", "f_c(g)", { name: "its fitness on case c" })),
})
