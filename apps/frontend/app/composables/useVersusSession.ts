import type { Bot, BotNetwork, Candidates, Move, Ply, Seat, Tally, VersusEngine, VersusStrategy } from "~/types/versus"

// One two-player session, whatever the game (docs/design/0006): two seats, each a human or a strategy,
// over a `VersusEngine`. Bots move on their own turn; a human builds a move one click at a time from
// the engine's own legal moves, so the UI can never offer a move the rules don't allow. Human vs bot,
// bot vs bot and pass-and-play are the same code with different seats. Runs on the main thread -- these
// games are cheap -- so the session *is* the game loop; a game that isn't would move `engine` into a
// worker behind the same interface.
//
// `createEngine` may be async (WebAssembly must load); `strategies` may change while the page is up
// (the run viewer swaps in another generation's champion under the same id) -- seats are rebuilt, the
// game in progress is not.
const MAX_ARENA_PLIES = 300

const same = (a: Move, b: Move) => a.length === b.length && a.every((c, i) => c === b[i])

export function useVersusSession<P>(
  createEngine: () => Promise<VersusEngine<P>>,
  strategies: MaybeRefOrGetter<VersusStrategy<P>[]>,
  options: { defaultSeats?: [Seat, Seat]; autoRestartMs?: number | null } = {},
) {
  const loading = ref(true)
  const error = ref<string | null>(null)
  let engine: VersusEngine<P> | null = null

  const seats = ref<[Seat, Seat]>(options.defaultSeats ?? ["human", "human"])
  const position = shallowRef<P | null>(null)
  const moves = shallowRef<Move[]>([])
  const currentPlayer = ref(0)
  const done = ref(false)
  const winner = ref<number | null>(null)
  const history = ref<Ply[]>([])
  // Diagnostics: what the bot to move is weighing now (`candidates`, chosen = null until it plays), and
  // every bot decision so far this game (`decisions`) -- only for bots that expose `scores()`.
  const candidates = shallowRef<Candidates | null>(null)
  // The same, per player: each seat keeps its latest, so a UI can show both side by side instead of one
  // panel that flips between players on every ply.
  const candidatesByPlayer = shallowRef<(Candidates | null)[]>([null, null])
  // Each seat's network, when its bot has one the diagram can draw.
  const networkByPlayer = shallowRef<(BotNetwork | null)[]>([null, null])
  const decisions = shallowRef<Candidates[]>([])
  const paused = ref(false)
  // The shared playback vocabulary (utils/playback.ts): a speed multiplier over a base pause between moves.
  const speed = ref(1)
  const botDelayMs = computed(() => Math.round(VERSUS_BASE_DELAY_MS / speed.value))
  const thinking = ref(false)
  // Watch pages replay: after a finished bot-vs-bot game, start another.
  const autoRestartMs = ref<number | null>(options.autoRestartMs ?? null)
  // Refs, not read off `engine` in a computed: the engine is a plain variable, so a computed would cache
  // the empty value from before it loaded.
  const players = ref<readonly string[]>([])
  const drawNote = ref("")
  const record = ref<{ wins: [number, number]; draws: number }>({ wins: [0, 0], draws: 0 })

  const strategyList = computed(() => toValue(strategies))
  const strategyById = (id: string) => strategyList.value.find((s) => s.id === id)
  const seatLabel = (player: number) => {
    const seat = seats.value[player]!
    return seat === "human" ? "Human" : (strategyById(seat)?.label ?? seat)
  }

  // The human's move under construction: cells picked so far (starts with the piece's cell).
  const path = ref<number[]>([])

  let bots: (Bot | null)[] = [null, null]
  let timer: ReturnType<typeof setTimeout> | null = null
  let seedCounter = Date.now() >>> 0
  const nextSeed = () => (seedCounter = (seedCounter + 0x9e3779b9) >>> 0)
  // Bumped whenever the game is replaced, so a pending bot move for the old one is dropped.
  let epoch = 0

  function refresh() {
    const e = engine!
    position.value = e.position()
    moves.value = e.legalMoves()
    currentPlayer.value = e.currentPlayer()
    done.value = e.isDone()
    winner.value = e.winner()
  }

  function rebuildBots() {
    bots.forEach((bot) => bot?.free())
    bots = seats.value.map((seat) => {
      const strategy = seat === "human" ? null : strategyById(seat)
      return strategy && engine ? strategy.create(engine, nextSeed()) : null
    })
    networkByPlayer.value = bots.map((bot) => bot?.network ?? null)
  }

  async function load() {
    try {
      engine = await createEngine()
      players.value = engine.players
      drawNote.value = engine.drawNote
      rebuildBots()
      refresh()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
    schedule()
  }

  // --- Human input ---------------------------------------------------------------------------------
  const humanTurn = computed(() => !done.value && seats.value[currentPlayer.value] === "human")
  /** Moves still possible given the cells picked so far. */
  const remainingMoves = computed(() => (path.value.length ? moves.value.filter((m) => path.value.every((c, i) => c === m[i])) : moves.value))
  /** Cells whose piece can start a move this turn. */
  const starts = computed<number[]>(() => (humanTurn.value ? [...new Set(moves.value.map((m) => m[0]!))] : []))
  /** Cells the picked piece can go to next. */
  const targets = computed<number[]>(() =>
    humanTurn.value && path.value.length ? [...new Set(remainingMoves.value.map((m) => m[path.value.length]).filter((c): c is number => c !== undefined))] : [],
  )

  /** A click on a board cell while it's a human's turn. */
  function clickCell(cell: number) {
    if (!humanTurn.value) return
    if (targets.value.includes(cell)) {
      const next = [...path.value, cell]
      const exact = moves.value.findIndex((m) => same(m, next))
      if (exact >= 0) play(exact)
      else path.value = next // a multi-jump with more landings to choose
    } else if (path.value.length <= 1) {
      // Pick a piece, switch to another, or click elsewhere to deselect. (Mid-chain the rules force the
      // same piece to keep going, so there is nothing to switch to.)
      path.value = starts.value.includes(cell) ? [cell] : []
    }
  }
  function cancelSelection() {
    path.value = []
  }

  // --- Playing --------------------------------------------------------------------------------------
  function play(index: number) {
    const e = engine
    const move = moves.value[index]
    if (!e || !move || done.value) return
    const player = e.currentPlayer()
    const text = e.notate(move)
    e.play(index)
    history.value = [...history.value, { player, move, text }]
    path.value = []
    refresh()
    if (done.value) recordResult()
    schedule()
  }

  function recordResult() {
    const w = winner.value
    if (w === null) record.value.draws++
    else record.value.wins[w] = (record.value.wins[w] ?? 0) + 1
  }

  function setLatest(c: Candidates) {
    const next = [...candidatesByPlayer.value]
    next[c.player] = c
    candidatesByPlayer.value = next
  }

  function schedule() {
    if (timer) clearTimeout(timer)
    timer = null
    thinking.value = false
    if (!engine) return
    const mine = epoch
    if (done.value) {
      if (autoRestartMs.value !== null && !paused.value && seats.value.every((s) => s !== "human")) {
        timer = setTimeout(() => mine === epoch && newGame(), autoRestartMs.value)
      }
      return
    }
    const bot = bots[currentPlayer.value]
    if (!bot) return
    // Shown while the bot "thinks" (and while paused): the options it is choosing between.
    const scores = bot.scores?.() ?? null
    candidates.value = scores ? { ply: history.value.length, player: currentPlayer.value, labels: moves.value.map((m) => engine!.notate(m)), scores, chosen: null } : null
    if (candidates.value) setLatest(candidates.value)
    if (paused.value) return
    thinking.value = true
    timer = setTimeout(() => {
      if (mine !== epoch || !engine || done.value) return
      const current = bots[currentPlayer.value]
      if (!current) return
      const index = current.pick()
      if (candidates.value && candidates.value.chosen === null && candidates.value.ply === history.value.length) {
        candidates.value = { ...candidates.value, chosen: index, activations: current.activations?.(index) ?? null }
        decisions.value = [...decisions.value, candidates.value]
        setLatest(candidates.value)
      }
      play(index)
    }, botDelayMs.value)
  }

  function newGame() {
    if (!engine) return
    epoch++
    engine.reset()
    history.value = []
    candidates.value = null
    candidatesByPlayer.value = [null, null]
    decisions.value = []
    path.value = []
    rebuildBots() // fresh seeds: the same two bots don't replay the same game
    refresh()
    schedule()
  }

  function resetRecord() {
    record.value = { wins: [0, 0], draws: 0 }
  }

  function setSeat(player: 0 | 1, seat: Seat, restart = false) {
    seats.value = player === 0 ? [seat, seats.value[1]] : [seats.value[0], seat]
    resetRecord()
    if (restart) return newGame()
    rebuildBots()
    path.value = []
    schedule()
  }

  function setPaused(value: boolean) {
    paused.value = value
    schedule()
  }

  /** Bot-vs-bot at full speed, off the visible board: `count` games between two strategies, seats
   *  alternating so first-move advantage cancels out. Yields to the page between games. Tally is from
   *  `a`'s point of view. */
  async function arena(aId: string, bId: string, count: number, onProgress: (played: number, tally: Tally) => void): Promise<Tally> {
    const a = strategyById(aId)!
    const b = strategyById(bId)!
    const tally: Tally = { wins: [0, 0], draws: 0 }
    for (let i = 0; i < count; i++) {
      const g = engine!.spawn()
      const aSeat = i % 2
      const bots = [a, b].map((s) => s.create(g, nextSeed()))
      const byPlayer = aSeat === 0 ? [bots[0]!, bots[1]!] : [bots[1]!, bots[0]!]
      for (let plies = 0; !g.isDone() && plies < MAX_ARENA_PLIES; plies++) g.play(byPlayer[g.currentPlayer()]!.pick())
      const w = g.winner()
      if (w === null) tally.draws++
      else tally.wins[w === aSeat ? 0 : 1]++
      bots.forEach((bot) => bot.free())
      g.dispose()
      onProgress(i + 1, { wins: [...tally.wins], draws: tally.draws })
      await new Promise((resolve) => setTimeout(resolve))
    }
    return tally
  }

  // Seats hold strategy *ids*; when the strategy list changes (a different generation's champion),
  // rebuild the bots but keep the game.
  watch(strategyList, () => {
    if (!engine) return
    rebuildBots()
    schedule()
  })
  watch(speed, schedule)
  watch(autoRestartMs, schedule)

  onScopeDispose(() => {
    epoch++
    if (timer) clearTimeout(timer)
    bots.forEach((bot) => bot?.free())
    engine?.dispose()
    engine = null
  })

  return {
    loading, error, seats, players, drawNote, candidates, candidatesByPlayer, networkByPlayer, decisions, position, moves, currentPlayer, done, winner, history, paused, speed, thinking, path, record, autoRestartMs,
    humanTurn, starts, targets, strategies: strategyList, seatLabel,
    load, clickCell, cancelSelection, newGame, setSeat, setPaused, resetRecord, arena, play,
  }
}

export type VersusSession<P> = ReturnType<typeof useVersusSession<P>>
