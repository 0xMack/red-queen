import { add, cases, formula, frac, term } from "~/utils/math/expr"

// Teaching a Snake's formulas (docs/design/0012): the shaped reward, the reward-hacking argument worked both ways
// (a step toward the food and a step back nets 0 with a symmetric nudge, −0.01 with the asymmetric one), and the
// headline improvement.

const closer = term("closer", "r_{\\text{closer}}", { name: "the reward for a step toward the food", num: { decimals: 2, max: 1, signed: true } })
const farther = term("farther", "r_{\\text{farther}}", { name: "the reward for a step away from it", num: { decimals: 2, max: 1, signed: true } })

export const reward = formula({
  id: "snake-reward",
  title: "Snake's shaped reward: one for food, minus one for dying, and a small nudge otherwise",
  lhs: term("r", "r", { symbol: "r", name: "one move's reward" }),
  body: cases(
    [term("eat", "+1", { name: "eating" }), "\\text{it eats}"],
    [term("die", "-1", { name: "dying", meaning: "Hitting a wall or itself -- or starving: too many moves without food." }), "\\text{it dies}"],
    [term("closer", "+0.01", { name: "a step toward the food" }), "\\text{it moves closer to the food}"],
    [term("farther", "-0.02", { name: "a step away from it", meaning: "Twice the reward for a step closer, on purpose: see below." }), "\\text{it moves farther}"],
  ),
})

const cycleRhs = add(closer, farther)

export const cycle = formula({
  id: "snake-cycle",
  title: "What a step toward the food and a step back nets: the reward for hovering in place",
  lhs: term("net", "r_{\\text{back and forth}}", {
    name: "the net reward of one step closer and one step back",
    meaning: "If this is 0, oscillating forever is a safe, costless strategy -- and evolution will find it.",
    num: { decimals: 2, max: 1, signed: true },
  }),
  body: cycleRhs,
  worked: cycleRhs,
  result: "net",
})

export const improvement = formula({
  id: "snake-improvement",
  title: "The improvement: best fitness after the fixes, over best fitness before",
  lhs: term("factor", "\\text{improvement}", { name: "how many times better", num: { decimals: 1, max: 99 } }),
  body: frac(term("after", "F_{\\text{after}}", { symbol: "fitness", name: "best fitness with 11 features and lexicase", num: { decimals: 2, max: 99 } }), term("before", "F_{\\text{before}}", { symbol: "fitness", name: "best fitness with the whole board", num: { decimals: 2, max: 99 } })),
  worked: frac(term("after", "F_{\\text{after}}", { symbol: "fitness", num: { decimals: 2, max: 99 } }), term("before", "F_{\\text{before}}", { symbol: "fitness", num: { decimals: 2, max: 99 } })),
  result: "factor",
})

