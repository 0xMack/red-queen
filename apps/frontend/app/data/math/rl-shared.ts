import { formula, paren, seq, term, type Formula } from "~/utils/math/expr"

// Pieces more than one RL chapter shows (docs/design/0012). Not a chapter's formulas itself -- /dev/math skips files
// that export none.

const MOVES = [
  ["q-left", "left", "turning left"],
  ["q-straight", "straight", "going straight"],
  ["q-right", "right", "turning right"],
] as const

/** The greedy move on a Snake demo board, worked: `a = argmax(Q(s,left), Q(s,straight), Q(s,right)) = straight`. The
 *  lab supplies `q-left`/`q-straight`/`q-right` (the values it chose from, sent with the move) and `move`
 *  (`\text{straight}`); its bars bind to the same ids. `q` is how one value is written (`Q(s,a)`, `Q(s,a;\theta)`). */
export function greedyChoice(id: string, q: (move: string) => string, what: string, max = 20): Formula {
  const value = (tid: string, move: string, doing: string) =>
    term(tid, q(`\\text{${move}}`), { symbol: "Q", name: `the value of ${doing}`, num: { decimals: 2, max, signed: true } })
  const [left, straight, right] = MOVES.map(([tid, move, doing]) => value(tid, move, doing))
  return formula({
    id,
    title: `The greedy move: the one with the largest value ${what}`,
    lhs: term("move", "a", { symbol: "a", name: "the move it plays" }),
    body: seq("\\operatorname*{arg\\,max}_{a}", term("q-row", q("a"), { symbol: "Q", name: `the three values ${what}` })),
    worked: seq("\\operatorname*{arg\\,max}", paren(seq(left!, ",\\;", straight!, ",\\;", right!))),
    result: "move",
  })
}

/** A lab's move and the values it chose from, as the greedy formula's values (none until the demo has played a move). */
export function greedyValues(values: readonly number[] | null | undefined, action: number | null | undefined): Record<string, number | string> {
  if (!values || values.length !== 3 || action === null || action === undefined) return {}
  return { "q-left": values[0]!, "q-straight": values[1]!, "q-right": values[2]!, move: `\\text{${MOVES[action]![1]}}` }
}

/** The bar for action `i` is that action's value, and part of the row. */
export const greedyBarTerms = (i: number): string[] => [MOVES[i]![0], "q-row"]
