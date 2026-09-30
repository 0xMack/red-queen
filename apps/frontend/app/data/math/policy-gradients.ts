import { add, bracket, formula, frac, mul, paren, seq, sub, term } from "~/utils/math/expr"
import { greedyChoice } from "./rl-shared"

// Policy Gradients' formulas (docs/design/0012). The theorem's weight follows the lab's rung of the ladder: the return
// (REINFORCE), the return minus a learned baseline, or the GAE advantage (A2C, PPO). The demo board's move is worked
// from the policy's own probabilities.

const gamma = term("gamma", "\\gamma", { symbol: "gamma" })
const lambda = term("lambda", "\\lambda", { symbol: "lambda" })
const logPi = term("grad-log-pi", "\\nabla_{\\theta} \\log \\pi_{\\theta}(a_t \\mid s_t)", {
  name: "the direction that makes this move more likely",
  meaning: "The gradient of the move's log-probability: step along it and the policy plays that move more often. For a softmax, 1 for the chosen move minus the probabilities.",
})
const ret = term("return", "G_t", { symbol: "G", name: "the return that followed" })
const value = (id: string, tex: string, name: string) => term(id, tex, { symbol: "V", name })

const weight = term("weight", ret, {
  name: "how good things turned out",
  meaning: "What each move's push is scaled by. The return is right on average and very noisy; subtracting a baseline, or using the critic, keeps it right with less noise.",
  expandable: true,
  forms: {
    baseline: { label: "minus a baseline", expr: sub(ret, value("baseline", "V(s_t)", "how good this situation usually is")) },
    gae: { label: "the GAE advantage", expr: term("advantage", "\\hat A_t", { symbol: "advantage" }) },
  },
})

export const theorem = formula({
  id: "pg-theorem",
  title: "The policy-gradient theorem: the gradient of the expected return is the average of each move's log-probability gradient, weighted by how good things turned out",
  lhs: seq("\\nabla_{\\theta}", term("J", "J(\\theta)", { symbol: "J" })),
  body: seq("\\mathbb{E}", bracket(mul(weight, logPi))),
})

export const tdResidual = formula({
  id: "pg-td",
  title: "One step's surprise: the reward plus gamma times the next state's value, minus this state's",
  lhs: term("delta", "\\delta_t", { symbol: "delta", name: "one step's surprise" }),
  body: sub(add(term("reward", "r_t", { symbol: "r" }), mul(gamma, value("v-next", "V(s_{t+1})", "the value of where it landed"))), value("v", "V(s_t)", "the value of where it was")),
})

export const gae = formula({
  id: "pg-gae",
  title: "Generalized advantage estimation: each step's surprise, plus gamma lambda times the advantage after it",
  lhs: term("advantage", "\\hat A_t", { symbol: "advantage" }),
  body: add(term("delta", "\\delta_t", { symbol: "delta", name: "this step's surprise" }), mul(gamma, lambda, term("advantage-next", "\\hat A_{t+1}", { symbol: "advantage", name: "the advantage one step later" }))),
})

const rho = term("ratio", "\\rho_t", { symbol: "ratio" })
const adv = term("advantage", "\\hat A_t", { symbol: "advantage" })
const eps = term("clip", "\\epsilon", { symbol: "clip" })

export const ratio = formula({
  id: "pg-ratio",
  title: "The probability ratio: how much more likely the move is now than when it was played",
  lhs: rho,
  body: frac(seq("\\pi_{\\theta}(a_t \\mid s_t)"), term("pi-old", "\\pi_{\\text{old}}(a_t \\mid s_t)", { symbol: "pi", name: "its probability when it was played" })),
})

export const ppo = formula({
  id: "pg-ppo",
  title: "PPO's objective: the smaller of the ratio times the advantage and the clipped ratio times the advantage",
  lhs: seq("L^{\\text{CLIP}}(\\theta)"),
  body: seq(
    "\\mathbb{E}",
    bracket(
      seq(
        "\\min",
        paren(
          seq(
            term("unclipped", mul(rho, adv), { name: "the plain push", meaning: "The advantage, scaled by how much likelier the move has become." }),
            ",\\;",
            term("clipped", mul(seq("\\operatorname{clip}", paren(seq(rho, ",\\, 1 -", eps, ",\\, 1 +", eps))), adv), {
              name: "the capped push",
              meaning: "The same, with the ratio held within 1 ± ε: past that, pushing further earns nothing.",
            }),
          ),
        ),
      ),
    ),
  ),
})

export const choice = greedyChoice("pg-choice", (a) => `\\pi(${a} \\mid s)`, "the policy gives", {
  max: 1,
  symbol: "pi",
  of: "the probability",
  plural: "probabilities",
  signed: false,
})
