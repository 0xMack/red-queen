import { add, bracket, formula, mul, seq, sub, term } from "~/utils/math/expr"

// Learning by Self-Play's formulas (docs/design/0012): the move a value network plays, the λ-return targets it learns
// from (with the sign flip between players), and the update. The lab's λ select is linked to λ.

const V = (id: string, tex: string, name: string, meaning?: string) => term(id, tex, { symbol: "V", name, meaning })
const lambda = term("lambda", "\\lambda", { symbol: "lambda", num: { decimals: 1, max: 1 } })

export const move = formula({
  id: "selfplay-move",
  title: "The move it plays: the one that leaves the position worst for the opponent",
  lhs: term("move", "a^*", { symbol: "a", name: "the move it plays" }),
  body: seq(
    "\\operatorname*{arg\\,max}_{a}",
    seq(
      "-",
      V("v-after", "V(s \\cdot a)", "the value of the position after the move", "Scored from the opponent's side -- it's their move next -- so the best move for me is the one that scores lowest for them."),
    ),
  ),
})

const target = (id: string, tex: string, name: string) => term(id, tex, { name })

export const lambdaReturn = formula({
  id: "selfplay-lambda-return",
  title: "Each position's target: a blend of the next position's value and its target, with the sign flipped because the next position is the other player's",
  lhs: target("g-t", "G^{\\lambda}_t", "this position's target"),
  body: seq(
    "-",
    bracket(
      add(
        mul(term("one-minus-lambda", sub("1", lambda), { name: "how much to trust the network's own estimate" }), V("v-next", "V(s_{t+1})", "the network's value of the next position")),
        mul(lambda, target("g-next", "G^{\\lambda}_{t+1}", "the next position's target: ultimately, the result")),
      ),
    ),
  ),
})

export const lambdaEnd = formula({
  id: "selfplay-end",
  title: "The last position's target is how the game actually ended, for the player who made the last move",
  lhs: target("g-last", "G^{\\lambda}_T", "the last position's target"),
  body: term("outcome", "z", {
    name: "the result",
    meaning: "+1 if the last mover won, −1 if they lost, 0 for a draw. The only reward in the whole game.",
  }),
})

export const update = formula({
  id: "selfplay-update",
  title: "One gradient step pulls every position's value toward its target",
  lhs: term("theta", "\\theta", { symbol: "weights" }),
  rel: "\\leftarrow",
  body: add(
    term("theta", "\\theta", { symbol: "weights" }),
    mul(
      term("alpha", "\\alpha", { symbol: "alpha" }),
      bracket(sub(target("g-t", "G^{\\lambda}_t", "this position's target"), V("v", "V(s_t)", "the network's value of this position"))),
      seq("\\nabla_{\\theta}", V("v", "V(s_t)", "the network's value of this position")),
    ),
  ),
})
