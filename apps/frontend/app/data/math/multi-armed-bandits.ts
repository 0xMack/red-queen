import { bracket, cases, formula, frac, mul, paren, seq, sqrt, term } from "~/utils/math/expr"

// Multi-Armed Bandits' formulas (docs/design/0012's pilot). Term ids are shared with the lab (BanditLab writes values
// under them after every pull) and with the table and tape (BanditTable, BanditTape mark themselves with them):
//   q-old / q-new  the pulled machine's estimate before / after the pull     n, reward, step, alpha
//   estimate       every machine's current estimate (the table's values)     count  every machine's pull count
//   bonus, c, t    UCB's exploration bonus and its parts                     epsilon, gamma, max-next, td

const two = (v: number) => v.toFixed(2)
// A payout: whole numbers as they are (a win pays 1), anything else to two places.
const payout = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2))

const n = term("n", "n", { symbol: "n", name: "pulls of this machine", format: String })
const reward = term("reward", "r", { symbol: "r", name: "what this pull paid", format: payout })
const qOld = term("q-old", "Q_n", {
  symbol: "Q",
  name: "the estimate before this pull",
  meaning: "The machine's average payout over its first n − 1 pulls.",
  format: two,
  expandable: true,
  forms: { average: { label: "as an average", expr: seq(frac("1", seq(n, "- 1")), "\\sum_{i=1}^{n-1} r_i") } },
})
const alpha = term("alpha", "\\alpha", { symbol: "alpha", format: two })
const step = term("step", frac("1", n), {
  symbol: "alpha",
  name: "the step",
  meaning: "How far to move toward the new payout. 1/n makes the estimate exactly the running average; a constant α forgets old payouts at a steady rate.",
  format: two,
  expandable: true,
  forms: { alpha: { label: "a constant step α", expr: alpha } },
})
const error = term("error", seq(reward, "-", qOld), {
  symbol: "delta",
  name: "the surprise",
  meaning: "How far the payout was from what the machine was expected to pay. The estimate moves a step along it.",
  format: two,
  expandable: true,
  forms: { delta: { label: "as δ", expr: "\\delta" } },
})

export const incremental = formula({
  id: "bandit-incremental",
  title: "The new estimate is the old one, moved a step of 1/n toward the payout",
  lhs: term("q-new", "Q_{n+1}", { symbol: "Q", name: "the estimate after this pull", format: two }),
  rel: "\\leftarrow",
  body: seq(qOld, "+", mul(step, paren(error))),
  worked: seq(qOld, "+", mul(step, paren(error))),
  result: "q-new",
})

const epsilon = term("epsilon", "\\varepsilon", { symbol: "epsilon", format: two })
const estimate = term("estimate", "Q(a)", { symbol: "Q", name: "each machine's estimate", meaning: "The values in the table: one per machine.", format: two })

export const epsilonGreedy = formula({
  id: "bandit-epsilon-greedy",
  title: "Epsilon-greedy: a random machine with probability epsilon, the best estimate otherwise",
  lhs: seq("a_t"),
  body: seq(
    cases(
      [term("explore", "\\text{any machine, at random}", { name: "explore", meaning: "Ignore the estimates: every machine is equally likely." }), seq("\\text{with probability }", epsilon)],
      [term("exploit", seq("\\operatorname*{arg\\,max}_a", estimate), { name: "exploit", meaning: "The machine with the best estimate so far." }), "\\text{otherwise}"],
    ),
  ),
})

const c = term("c", "c", { symbol: "c", format: two })
const t = term("t", "t", { symbol: "t", name: "pulls so far", format: String })
const count = term("count", "N(a)", { symbol: "N", name: "this machine's pulls", format: String })
const bonus = term("bonus", mul(c, sqrt(frac(seq("\\ln", t), count))), {
  name: "the exploration bonus",
  meaning: "How much better the machine could be, for all the player knows: large for a machine barely tried, shrinking as its pulls add up.",
  format: two,
})

export const ucb = formula({
  id: "bandit-ucb",
  title: "UCB: pull the machine whose estimate plus uncertainty bonus is highest",
  lhs: seq("a_t"),
  body: seq("\\operatorname*{arg\\,max}_a", bracket(term("score", seq(estimate, "+", bonus), { name: "the machine's score", format: two }))),
  // The worked row is the score of the machine just pulled: `score(C) = 0.89 + 1.41·√(ln 27 / 9) = 1.74`.
  workedLhs: seq(
    "\\text{score}(",
    term("arm", "a", { symbol: "a", name: "the machine pulled", meaning: "The machine this pull chose: the one with the highest score." }),
    ")",
  ),
  worked: seq(estimate, "+", bonus),
  result: "score",
})

export const thompson = formula({
  id: "bandit-thompson",
  title: "Thompson sampling: draw a plausible win rate for every machine and pull the highest draw",
  body: seq(
    term("theta", "\\theta_a", { symbol: "theta" }),
    "\\sim \\operatorname{Beta}",
    paren(
      seq(
        "1 +",
        term("wins", "w_a", { name: "wins", meaning: "How many of the machine's pulls paid." }),
        ",\\; 1 +",
        term("losses", "\\ell_a", { name: "losses", meaning: "How many of its pulls paid nothing." }),
      ),
    ),
    ",\\qquad a_t = \\operatorname*{arg\\,max}_a \\theta_a",
  ),
})

const gamma = term("gamma", "\\gamma", { symbol: "gamma", format: (v) => String(v) })
const qSa = term("q-old", "Q(s,a)", { symbol: "Q", name: "this machine's value in this room, before the pull", format: two })
const maxNext = term("max-next", "\\max_{a'} Q(s',a')", {
  name: "the best value of the room it leads to",
  meaning: "The best value in the row of the room this pull led to -- what the future looks like from there. 0 when the game is over.",
  format: two,
})
const td = term("td", seq(term("reward", "r", { symbol: "r", name: "what this pull paid", format: payout }), "+", mul(gamma, maxNext), "-", qSa), {
  symbol: "delta",
  name: "the TD error",
  meaning: "The target -- the payout plus the discounted best value of where the pull leads -- minus the current estimate.",
  format: two,
  expandable: true,
  forms: { delta: { label: "as δ", expr: "\\delta" } },
})

export const bellman = formula({
  id: "bandit-bellman",
  title: "Q-learning's update: move the value toward the payout plus gamma times the best value of the next room",
  lhs: term("q-new", "Q(s,a)", { symbol: "Q", name: "the value after the pull", format: two }),
  rel: "\\leftarrow",
  body: seq(qSa, "+", mul(alpha, bracket(td))),
  worked: seq(qSa, "+", mul(alpha, bracket(td))),
  result: "q-new",
})
