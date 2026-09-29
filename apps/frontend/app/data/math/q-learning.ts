import { add, bracket, formula, mul, sub, term } from "~/utils/math/expr"
import { greedyChoice } from "./rl-shared"

// Q-learning's formulas (docs/design/0012). The update rule is shown by the lab (QLearningLab's `formula` slot) with
// its knobs linked -- α, γ, ε and n are sliders, and the algorithm switch picks the target's form (Q-learning's max,
// SARSA's next move, or the n-step return). The lab trains in a worker at thousands of steps a second, so the update
// stays symbolic; the greedy choice on the demo board is shown worked: the row's three values and the move they pick.

const gamma = term("gamma", "\\gamma", { symbol: "gamma", num: { decimals: 2, max: 1 } })
const alpha = term("alpha", "\\alpha", { symbol: "alpha", num: { decimals: 2, max: 1 } })
const reward = (id: string, tex: string, name: string) => term(id, tex, { symbol: "r", name })

// --- The return ------------------------------------------------------------------------------------------------------

const returnSum = term(
  "return-sum",
  add(
    reward("r1", "r_{t+1}", "the next reward"),
    mul(gamma, reward("r2", "r_{t+2}", "the reward after that")),
    mul(term("gamma2", "\\gamma^2", { name: "γ, twice", meaning: "Two steps away, a reward is discounted twice: at γ = 0.95, to 0.90." }), reward("r3", "r_{t+3}", "three steps away")),
    "\\cdots",
  ),
  {
    name: "every future reward, discounted",
    meaning: "Each reward counts γ times less than the one before it: a reward twenty moves away, at γ = 0.95, counts about a third.",
    expandable: true,
    forms: {
      recursive: {
        label: "recursively",
        expr: add(reward("r1", "r_{t+1}", "the next reward"), mul(gamma, term("return-next", "G_{t+1}", { symbol: "G", name: "the return from the next step" }))),
      },
    },
  },
)

export const discountedReturn = formula({
  id: "q-return",
  title: "The return: every future reward, each discounted by gamma once more than the last",
  lhs: term("return", "G_t", { symbol: "G", name: "the return from step t" }),
  body: returnSum,
})

// --- The update ------------------------------------------------------------------------------------------------------

const q = term("q-sa", "Q(s,a)", { symbol: "Q", name: "the table's value for this move, here", num: { decimals: 2, max: 20, signed: true } })
const maxNext = term("max-next", "\\max_{a'} Q(s',a')", {
  name: "the best value from where it lands",
  meaning: "Q-learning's guess about the future: the best move's value in the next situation, whatever it will actually do there.",
})
const target = term("target", add(term("reward", "r", { symbol: "r", name: "this move's reward" }), mul(gamma, maxNext)), {
  name: "the target",
  meaning: "What one step of real experience says the value should be: the reward, plus the discounted value of where the move led.",
  expandable: true,
  forms: {
    sarsa: {
      label: "SARSA: the move it will make",
      expr: add(term("reward", "r", { symbol: "r" }), mul(gamma, term("q-next", "Q(s',a')", { symbol: "Q", name: "the value of the move it will actually make next", meaning: "SARSA bootstraps from its real next move, exploration included -- the value of the way it actually plays." }))),
    },
    nstep: {
      label: "n-step",
      expr: add(
        term("rewards-n", "r_1 + \\gamma r_2 + \\cdots + \\gamma^{n-1} r_n", { name: "n real rewards", meaning: "Wait n moves and use the rewards that actually came, before trusting the table." }),
        mul(term("gamma-n", "\\gamma^{n}", { name: "γ, n times" }), term("max-n", "\\max_{a'} Q(s_n,a')", { name: "the best value n moves on" })),
      ),
    },
  },
})
const td = term("td", sub(target, q), {
  symbol: "delta",
  name: "the TD error",
  meaning: "How surprised the table was: the target minus what it predicted.",
  expandable: true,
  forms: { delta: { label: "as δ", expr: "\\delta" } },
})

export const update = formula({
  id: "q-update",
  title: "Q-learning's update: move the value a fraction alpha of the way to the reward plus gamma times the best next value",
  lhs: term("q-new", "Q(s,a)", { symbol: "Q", name: "the value after the update" }),
  rel: "\\leftarrow",
  body: add(q, mul(alpha, bracket(td))),
})

// --- The greedy choice (the demo board) ------------------------------------------------------------------------------

export const greedy = greedyChoice("q-greedy", (a) => `Q(s,${a})`, "in the current row")
