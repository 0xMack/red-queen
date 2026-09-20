import type { VersusSession } from "~/composables/useVersusSession"
import { squareOf, type CheckersPosition } from "~/utils/checkersEngine"

// What a Checkers board shows about the game in progress, derived from a versus session: the position, the last
// move's path (the board animates it and draws its trail), whose turn it glows for, and the result banner. Both
// Checkers stages (the game page and the run viewer) draw the same board from the same session, so this lives
// once.
export function useCheckersBoardView(session: VersusSession<CheckersPosition>) {
  const { position, history, currentPlayer, done, winner, players, drawNote } = session

  const board = computed(() => position.value ?? { width: 8, height: 8, cells: [] })
  const trail = computed(() => history.value.at(-1)?.move.map(squareOf) ?? [])
  const turn = computed<0 | 1 | null>(() => (done.value ? null : (currentPlayer.value as 0 | 1)))
  const banner = computed(() => {
    if (!done.value) return null
    const w = winner.value
    if (w === null) return { title: "Draw", sub: drawNote.value, tone: "draw" as const }
    return { title: `${players.value[w]} wins`, sub: session.seatLabel(w), tone: w === 0 ? ("red" as const) : ("black" as const) }
  })
  const pieceCounts = computed(() => {
    const count = [0, 0]
    for (const cell of board.value.cells) count[cell.label.startsWith("red") ? 0 : 1]!++
    return count
  })

  return { board, trail, turn, banner, pieceCounts }
}
