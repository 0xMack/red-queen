// The game-agnostic contract behind every two-player page (docs/design/0006): a *session* (seats, human
// input, bot turns, history, an arena) drives any *engine* (one game's rules) with any *strategies*.
// Checkers is the first engine (utils/checkers.ts); the next game supplies an engine, its strategies
// and a board, and gets the whole stage -- see composables/useVersusSession.ts and VersusStage.vue.

/** A move is a whole turn as the board cells it touches, in order (cell = y * width + x): checkers'
 *  from + each landing square, a chess move's from + to. The human builds it one click at a time. */
export type Move = number[]

/** Someone who picks moves: an index into the engine's *current* `legalMoves()`. Owns whatever it
 *  needs (a seeded PRNG, WebAssembly memory), hence `free()`. */
/** The network a bot evaluates positions with, for the diagram to draw: a fixed-topology one (evolve.WeightVector's
 *  flat layout, tanh layers) or an evolved graph (a NEAT genome). Absent for a bot with no network (a material
 *  search, random). */
export type BotNetwork =
  | {
      kind: "layered"
      weights: readonly number[]
      layerSizes: readonly number[]
      inputLabels?: readonly string[]
      outputLabels?: readonly string[]
    }
  | {
      kind: "graph"
      genome: import("~/utils/neat").Genome
      inputLabels?: readonly string[]
      outputLabels?: readonly string[]
    }

/** What lights a network diagram up: each layer's values (layered), or every live node's value by node id (graph). */
export type Activations = number[][] | Map<number, number>

export interface Bot {
  pick(): number
  /** The network the bot evaluates positions with, if it has one the diagram can draw. */
  network?: BotNetwork
  /** The network's values when it evaluated the position that legal move `moveIndex` leads to -- what lights the
   *  diagram up -- or null. */
  activations?(moveIndex: number): Activations | null
  /** What the bot thinks each legal move is worth (higher is better, from its own side), in
   *  `legalMoves()` order -- its reasoning made visible -- or null for a bot that doesn't score moves
   *  (a random one). The diagnostics panels plot this; a strategy that exposes it gets them for free. */
  scores?(): number[] | null
  free(): void
}

export interface VersusStrategy<P> {
  id: string
  label: string
  description: string
  /** A new player for `engine`'s game. `seed` makes its tie-breaks reproducible. */
  create(engine: VersusEngine<P>, seed: number): Bot
}

/** One game's rules, as the session needs them. `P` is the position a board renders. */
export interface VersusEngine<P> {
  /** Display names of the seats, in player order ("Red", "Black"). */
  readonly players: readonly string[]
  /** Why a game with no winner ended ("40 moves without a capture"). */
  readonly drawNote: string
  reset(): void
  /** Current legal moves in the engine's own order; a bot's pick indexes this list. Empty once done. */
  legalMoves(): Move[]
  currentPlayer(): number
  /** Apply `legalMoves()[index]`. */
  play(index: number): void
  isDone(): boolean
  /** The winning seat, or null while ongoing or drawn (use `isDone()` to tell them apart). */
  winner(): number | null
  position(): P
  notate(move: Move): string
  /** An independent fresh game of the same kind -- the arena plays on these, off the visible board. */
  spawn(): VersusEngine<P>
  dispose(): void
}

export type Seat = "human" | string // "human", or a strategy id

export interface Ply {
  player: number
  move: Move
  text: string
}

/** What one bot weighed at one ply: every legal move, its score, and (once played) which it chose. */
export interface Candidates {
  ply: number
  player: number
  labels: string[] // each legal move, in the engine's notation
  scores: number[]
  chosen: number | null
  /** The network's activations on the position the chosen move leaves behind (bots with a network only). */
  activations?: Activations | null
}

export interface Tally {
  wins: [number, number] // [a's, b's]
  draws: number
}
