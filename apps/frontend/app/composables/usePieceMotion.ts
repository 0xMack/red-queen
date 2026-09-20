// Board-piece motion, for any grid game whose engine gives pieces a stable identity: diff each new position
// against the last one and replay the difference as motion instead of a teleport -- the moved piece hops along
// the move's path square by square, a piece it jumps leaves as the mover passes over it, a crowned piece
// flashes. It knows nothing about Checkers beyond "a jump is two files, and the jumped piece sits between".
//
// Two things this must get right, both of which broke the first version (a piece "disappearing after moving
// and re-animating in from the wrong direction"):
// - **Render order is by id, never the engine's order.** An engine may list a moved piece last; a keyed list
//   rendered in that order makes Vue physically *move the DOM node* to match, and moving an SVG node restarts its CSS
//   -- the glide is lost and the entrance animation (with its staggered, invisible delay) replays.
// - **A snap is a new scene**, so every piece gets a fresh key (`epoch`). Ids restart at 0 with a new game;
//   without the epoch, surviving keys would reuse the old elements and glide them across the board from wherever the
//   last game left them.
//
// Rules of the road:
// - A normal move changes exactly one piece's position. Anything else (a new game, a jump to another
//   position) can't be animated as a move, so the board *snaps* -- and pieces then mount with their entrance
//   animation (CheckersPiece).
// - If the next position arrives while a move is still animating (fast bots, a small delay), the animation
//   in flight is dropped and the board snaps to where that move ended before starting the new one, so the
//   display never falls behind the game.
// - Everything is timers on plain state, so `prefers-reduced-motion` is handled by the CSS that consumes
//   `hopping` / `leaving` / `crowned` (they simply have no visual effect), and the timings live here.

export interface MotionPiece {
  id: number | string
  x: number
  y: number
  label: string
}

/** A piece as drawn: its *displayed* square (which lags the game's during a hop) plus what it is doing. */
export interface ShownPiece extends MotionPiece {
  /** The v-for key: id + the scene it belongs to (see `epoch` below), so a new scene remounts instead of gliding. */
  key: string
  /** Mid-hop: lifted off the board. */
  hopping?: boolean
  /** Being captured: shrinking away. */
  leaving?: boolean
  /** Just became a king: play the crowning flash. */
  crowned?: boolean
}

export interface Burst {
  key: number
  x: number
  y: number
}

export interface PieceMotionOptions {
  /** Time to hop one square (ms). */
  hopMs?: number
  /** How long a captured piece takes to leave (ms). */
  leaveMs?: number
  /** How long the crowning flash lasts (ms). */
  crownMs?: number
}

const isKing = (label: string) => label.endsWith("king")

export function usePieceMotion(pieces: () => MotionPiece[], trail: () => [number, number][], options: PieceMotionOptions = {}) {
  const { hopMs = 230, leaveMs = 320, crownMs = 900 } = options
  let epoch = 0
  const keyed = (p: MotionPiece): ShownPiece => ({ ...p, key: `${epoch}:${p.id}` })
  // Always stored in id order (see the header) -- never the engine's.
  const byId = (a: MotionPiece, b: MotionPiece) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
  const ordered = (list: ShownPiece[]) => [...list].sort(byId)
  const shown = shallowRef<ShownPiece[]>(ordered(pieces().map(keyed)))
  const bursts = shallowRef<Burst[]>([])
  /** A move is still being replayed. Anything that should wait for the board to be still (a result banner) reads this. */
  const busy = ref(false)
  let timers: ReturnType<typeof setTimeout>[] = []
  let burstKey = 0

  const later = (ms: number, fn: () => void) => timers.push(setTimeout(fn, ms))
  const cancel = () => {
    timers.forEach(clearTimeout)
    timers = []
    busy.value = false
  }
  const patch = (id: MotionPiece["id"], change: Partial<ShownPiece>) => {
    shown.value = ordered(shown.value.map((p) => (p.id === id ? { ...p, ...change } : p)))
  }

  watch(pieces, (next, prev) => {
    cancel()
    // A scene change: new epoch, so every piece is a fresh element (and pops in) rather than a survivor gliding.
    const snap = () => {
      epoch++
      shown.value = ordered(next.map(keyed))
    }
    if (!prev) return snap()
    try {
      plan(next, prev, snap)
    } catch (error) {
      // Motion is decoration: if planning it ever fails, show the position rather than break the page.
      console.warn("usePieceMotion: showing the position without animating", error)
      cancel()
      snap()
    }
  })

  function plan(next: MotionPiece[], prev: MotionPiece[], snap: () => void) {

    const before = new Map(prev.map((p) => [p.id, p]))
    const after = new Map(next.map((p) => [p.id, p]))
    const moved = next.filter((p) => {
      const was = before.get(p.id)
      return was && (was.x !== p.x || was.y !== p.y)
    })
    const path = trail()
    const mover = moved[0]
    // The path must start on the mover's old square and end on its new one. (An empty or one-square trail --
    // right after a new game -- explains nothing, and must not be indexed: an exception in this watcher aborts
    // the *component's* update and leaves Vue's renderer in a broken state, freezing the page.)
    const first = path[0]
    const last = path.at(-1)
    const startsRight = !!mover && path.length >= 2 && !!first && first[0] === before.get(mover.id)!.x && first[1] === before.get(mover.id)!.y
    const endsRight = !!mover && !!last && last[0] === mover.x && last[1] === mover.y
    // Not something a single move explains (a new game, a jump in the position): snap instead of guessing.
    if (next.length > prev.length || moved.length !== 1 || !startsRight || !endsRight) return snap()

    // Start from where the last move ended -- the previous animation may have been cut short.
    shown.value = ordered(prev.map(keyed))
    busy.value = true
    const removed = prev.filter((p) => !after.has(p.id))
    const captured = new Set<MotionPiece["id"]>()

    path.slice(1).forEach(([x, y], hop) => {
      const at = hop * hopMs
      const [fromX, fromY] = path[hop]!
      later(at, () => patch(mover!.id, { x, y, hopping: true }))
      if (Math.abs(x - fromX) === 2) {
        // A jump: the piece between the two squares goes as the mover passes over it.
        const midX = (x + fromX) / 2
        const midY = (y + fromY) / 2
        const victim = removed.find((p) => p.x === midX && p.y === midY)
        if (victim) {
          captured.add(victim.id)
          const leaveAt = at + hopMs * 0.55
          later(leaveAt, () => {
            patch(victim.id, { leaving: true })
            bursts.value = [...bursts.value, { key: ++burstKey, x: midX, y: midY }]
            const key = burstKey
            later(leaveMs + 250, () => (bursts.value = bursts.value.filter((b) => b.key !== key)))
          })
          // Gone only once its exit has played -- even if the mover has already landed by then.
          later(leaveAt + leaveMs, () => (shown.value = ordered(shown.value.filter((p) => p.id !== victim.id))))
        }
      }
    })

    const end = (path.length - 1) * hopMs
    // Landed: settle on the game's position, keeping captured pieces on the board just long enough to finish
    // leaving, and crown if that's what happened.
    later(end + 20, () => {
      const crowned = !isKing(before.get(mover!.id)!.label) && isKing(mover!.label)
      const leavers = shown.value.filter((p) => captured.has(p.id))
      shown.value = ordered([...next.map((p) => (p.id === mover!.id ? { ...keyed(p), crowned } : keyed(p))), ...leavers])
      if (crowned) later(crownMs, () => patch(mover!.id, { crowned: false }))
      // Still while the last captured piece finishes leaving; not while a crown flash plays (that's ornament).
      later(leavers.length ? leaveMs : 0, () => (busy.value = false))
    })
  }

  onScopeDispose(cancel)
  return { shown, bursts, busy }
}
