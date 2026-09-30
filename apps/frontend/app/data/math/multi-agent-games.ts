import { cases, formula, frac, seq, sub, term } from "~/utils/math/expr"

// Multi-Agent Games' formulas (docs/design/0012): the material count the hand-written players score by (the board32
// encoding: +1 per man, +2 per king, the opponent's negative), the one- and two-move material scores, and the match
// fitness an evolved player is scored by.

export const material = formula({
  id: "mag-material",
  title: "Material: the sum over the board, men counting 1 and kings 2, the opponent's pieces negative",
  lhs: seq(term("m", "m", { name: "material", meaning: "The piece balance from one side: positive when that side is ahead." }), "(s)"),
  body: seq(
    "\\sum_{\\text{squares}}",
    term("piece", "v", {
      name: "a square's value",
      meaning: "+1 for your man, +2 for your king, −1 and −2 for the opponent's, 0 when empty -- the board32.v1 encoding a network also reads.",
    }),
  ),
})

const after = (id: string, tex: string, name: string, meaning?: string) => term(id, tex, { name, meaning })

export const material1 = formula({
  id: "mag-material-1",
  title: "Material-1: a move is as good as the material it leaves, from my side",
  lhs: seq("\\text{score}_1(a)"),
  body: seq(
    "-\\,m",
    "\\big(",
    after("after-a", "s \\cdot a", "the position after my move", "Encoded from the opponent's side -- it's their move -- so my material is its negative."),
    "\\big)",
  ),
})

export const material2 = formula({
  id: "mag-material-2",
  title: "Material-2: a move is as good as the worst material the opponent's best reply leaves me",
  lhs: seq("\\text{score}_2(a)"),
  body: cases(
    [term("win", "W", { name: "a win", meaning: "A move that ends the game in my favour beats every material count." }), "\\text{if } a \\text{ wins}"],
    [
      seq(
        term("min-reply", "\\min_{a'}", { name: "the opponent's best reply", meaning: "Assume they answer with whatever is worst for me: the two-move lookahead that wins 190 of 200 games." }),
        "m\\big(",
        after("after-reply", "s \\cdot a \\cdot a'", "the position after their reply"),
        "\\big)",
      ),
      "\\text{otherwise}",
    ],
  ),
})

export const matchFitness = formula({
  id: "mag-match-fitness",
  title: "Match fitness: the mean outcome over every reference opponent, played from both seats",
  lhs: term("fitness", "F(g)", { symbol: "fitness", name: "the genome's match fitness" }),
  body: seq(
    frac("1", term("games", "2|O|", { name: "games played", meaning: "Every reference opponent, from both seats: first-move advantage cancels out." })),
    "\\sum_{o \\in O} \\sum_{\\text{seat}}",
    term("outcome", "z", { name: "the result", meaning: "+1 for a win, 0 for a draw, −1 for a loss: one 'test case' per (opponent, seat), so lexicase selection works unchanged." }),
  ),
})

// The fitness gain the chapter reports, worked.
export const improvement = formula({
  id: "mag-improvement",
  title: "The improvement over 40 generations",
  lhs: term("gain", "\\Delta F", { name: "the gain in mean fitness", num: { decimals: 2, max: 9, signed: true } }),
  body: sub(term("f-end", "\\bar F_{40}", { name: "mean fitness after 40 generations", num: { decimals: 2, max: 9 } }), term("f-start", "\\bar F_{0}", { name: "mean fitness at the start", num: { decimals: 2, max: 9 } })),
  worked: sub(term("f-end", "\\bar F_{40}", { name: "mean fitness after 40 generations", num: { decimals: 2, max: 9 } }), term("f-start", "\\bar F_{0}", { name: "mean fitness at the start", num: { decimals: 2, max: 9 } })),
  result: "gain",
})
