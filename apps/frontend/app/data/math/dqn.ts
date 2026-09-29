import { add, cases, formula, frac, mul, paren, seq, sub, term } from "~/utils/math/expr"
import { greedyChoice } from "./rl-shared"

// Deep Q-Networks' formulas (docs/design/0012). The target and the loss follow the lab's switches: with the target
// network off, θ⁻ is θ itself; Double DQN lets the online network choose the next move and the target network value
// it; 3-step returns wait for three real rewards; the replay buffer is where the minibatch is drawn from.

const theta = term("theta", "\\theta", { symbol: "weights", name: "the online network's weights" })
const thetaMinus = term("theta-minus", "\\theta^-", {
  symbol: "weightsTarget",
  expandable: true,
  forms: { online: { label: "no target network: θ itself", expr: "\\theta" } },
})
const gamma = term("gamma", "\\gamma", { symbol: "gamma" })

const target = term("target", add(term("reward", "r", { symbol: "r" }), mul(gamma, seq("\\max_{a'} Q(s',a';", thetaMinus, ")"))), {
  name: "the target",
  meaning: "What the network is pulled toward: the reward, plus γ times the best next value -- computed by the frozen target network, and held constant.",
  expandable: true,
  forms: {
    double: {
      label: "Double DQN",
      expr: add(
        term("reward", "r", { symbol: "r" }),
        mul(gamma, seq("Q\\big(s',", term("double-pick", "\\operatorname*{arg\\,max}_{a'} Q(s',a';\\theta)", { name: "the move the online network picks", meaning: "Double DQN: the online network chooses the next move, the target network says what it's worth. The max of noisy estimates is biased upward; splitting choosing from valuing removes most of it." }), ";", thetaMinus, "\\big)")),
      ),
    },
    nstep: {
      label: "3-step returns",
      expr: add(term("rewards-3", "r_1 + \\gamma r_2 + \\gamma^2 r_3", { name: "three real rewards" }), mul(term("gamma-3", "\\gamma^3", { name: "γ, three times" }), seq("\\max_{a'} Q(s_3,a';", thetaMinus, ")"))),
    },
  },
})

export const dqnTarget = formula({
  id: "dqn-target",
  title: "The target: the reward plus gamma times the best next value, according to the target network",
  lhs: term("y", "y", { name: "the target", meaning: "What Q(s, a; θ) should have said." }),
  body: target,
})

const delta = term("delta", sub(term("y", "y", { name: "the target" }), seq("Q(s,a;", theta, ")")), {
  symbol: "delta",
  name: "the TD error",
  expandable: true,
  forms: { short: { label: "as δ", expr: "\\delta" } },
})
const huber = term("huber", seq("\\operatorname{Huber}", paren(delta)), {
  name: "the Huber loss",
  meaning:
    "Squared error near the target, absolute error far from it: a big surprise gives a gradient of at most 1, so one wild target can't throw the weights far.",
  expandable: true,
  forms: {
    cases: {
      label: "written out",
      expr: cases([seq(frac("1", "2"), "\\delta^2"), "|\\delta| \\le 1"], ["|\\delta| - \\tfrac{1}{2}", "\\text{otherwise}"]),
    },
  },
})

export const dqnLoss = formula({
  id: "dqn-loss",
  title: "The loss: the Huber loss of the TD error, averaged over a random minibatch from the replay buffer",
  lhs: seq(term("loss", "L", { symbol: "loss" }), "(", theta, ")"),
  body: seq(
    term("replay", "\\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{D}}", {
      name: "the average over a random minibatch",
      meaning: "Experience replay: every move goes into a buffer of the last 50,000, and each update trains on 32 drawn at random -- not on the moves the snake just made, which are nearly all the same situation.",
    }),
    huber,
  ),
})

export const dqnStep = formula({
  id: "dqn-step",
  title: "Each update moves the weights a small step down the gradient of the loss",
  lhs: theta,
  rel: "\\leftarrow",
  body: sub(theta, mul(term("eta", "\\eta", { symbol: "eta" }), seq("\\nabla_{\\theta}", term("loss", "L", { symbol: "loss" })))),
})

export const greedy = greedyChoice("dqn-greedy", (a) => `Q(s,${a};\\theta)`, "the network computes", 100)
