import { formula, paren, seq, term, type Formula } from "~/utils/math/expr"

// Pieces more than one RL chapter shows (docs/design/0012). Not a chapter's formulas itself -- /dev/math skips files
// that export none.

const MOVES = [
  ["q-left", "left", "turning left"],
  ["q-straight", "straight", "going straight"],
  ["q-right", "right", "turning right"],
] as const

/** The chosen move on a Snake demo board, worked: `a = argmax(Q(s,left), Q(s,straight), Q(s,right)) = straight`. The
 *  lab supplies `q-left`/`q-straight`/`q-right` (the numbers it chose from, sent with the move) and `move`
 *  (`\text{straight}`); its bars bind to the same ids. `q` writes one of the numbers (`Q(s,a)`, `\pi(a \mid s)`), `symbol`
 *  and `of` say what they are (values by default; probabilities for a policy). */
export function greedyChoice(
  id: string,
  q: (move: string) => string,
  what: string,
  opts: { max?: number; symbol?: string; of?: string; plural?: string; signed?: boolean } = {},
): Formula {
  const { max = 20, symbol = "Q", of = "the value", plural = "values", signed = true } = opts
  const value = (tid: string, move: string, doing: string) =>
    term(tid, q(`\\text{${move}}`), { symbol, name: `${of} of ${doing}`, num: { decimals: 2, max, signed } })
  const [left, straight, right] = MOVES.map(([tid, move, doing]) => value(tid, move, doing))
  return formula({
    id,
    title: `The move it plays: the one with the largest ${of.replace(/^the /, "")} ${what}`,
    lhs: term("move", "a", { symbol: "a", name: "the move it plays" }),
    body: seq("\\operatorname*{arg\\,max}_{a}", term("q-row", q("a"), { symbol, name: `the three ${plural} ${what}` })),
    worked: seq("\\operatorname*{arg\\,max}", paren(seq(left!, ",\\;", straight!, ",\\;", right!))),
    result: "move",
  })
}

/** A lab's move and the numbers it chose from, as the greedy formula's values (none until the demo has played). */
export function greedyValues(values: readonly number[] | null | undefined, action: number | null | undefined): Record<string, number | string> {
  if (!values || values.length !== 3 || action === null || action === undefined) return {}
  return { "q-left": values[0]!, "q-straight": values[1]!, "q-right": values[2]!, move: `\\text{${MOVES[action]![1]}}` }
}

/** The bar for action `i` is that action's number, and part of the row. */
export const greedyBarTerms = (i: number): string[] => [MOVES[i]![0], "q-row"]
