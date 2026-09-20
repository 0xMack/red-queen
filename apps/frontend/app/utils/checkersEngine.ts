import init, { CheckersGame, CheckersStrategy } from "~/wasm/games/games.js"
import gamesWasmUrl from "~/wasm/games/games_bg.wasm?url"
import { activations as neatActivations, genomeFromJson, type Genome, type GenomeJson } from "~/utils/neat"
import type { Bot, BotNetwork, Move, VersusEngine, VersusStrategy } from "~/types/versus"

// Checkers as a `VersusEngine` (docs/design/0006): the Rust rules compiled to WebAssembly
// (app/wasm/games -- the same core training plays through PyO3) behind the game-agnostic interface, and
// the Rust strategies (rust/core/src/checkers_strategies.rs) as `VersusStrategy`s. Nothing here
// re-implements a rule or a player: every legal move, every bot decision, every tie-break comes from
// the core, identical to what jobs/ and the evaluation see for the same seed.

/** One piece on the board. `id` is stable for the piece's whole life -- through moves and crowning -- so a
 *  board can animate *that piece* gliding rather than one disappearing and another appearing. */
export interface CheckersPiece {
  id: number
  x: number
  y: number
  label: "red_man" | "red_king" | "black_man" | "black_king"
}

export interface CheckersPosition {
  width: number
  height: number
  cells: CheckersPiece[]
}

export const BOARD_SIZE = 8
const MAX_MOVES_WITHOUT_CAPTURE = 40
const PIECES = ["red_man", "red_king", "black_man", "black_king"] as const

export const cellOf = (x: number, y: number) => y * BOARD_SIZE + x
export const squareOf = (cell: number): [number, number] => [cell % BOARD_SIZE, Math.floor(cell / BOARD_SIZE)]
export const squareName = (cell: number) => {
  const [x, y] = squareOf(cell)
  return `${"abcdefgh"[x]}${y + 1}`
}
/** A move's jumps: each consecutive pair two files apart captures the piece between them. */
export const isJump = (from: number, to: number) => Math.abs(squareOf(to)[0] - squareOf(from)[0]) === 2
export const jumpedSquare = (from: number, to: number) => {
  const [fx, fy] = squareOf(from)
  const [tx, ty] = squareOf(to)
  return cellOf((fx + tx) / 2, (fy + ty) / 2)
}

/** The 32 playable squares, in the observation's order (`games.checkers`: row by row from Red's side) --
 *  what the evaluator network's inputs are called. */
export const INPUT_LABELS: readonly string[] = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, cell) => cell)
  .filter((cell) => (squareOf(cell)[0] + squareOf(cell)[1]) % 2 === 1)
  .map(squareName)

let ready: Promise<void> | null = null
export function ensureCheckersReady(): Promise<void> {
  ready ??= init({ module_or_path: gamesWasmUrl }).then(() => undefined)
  return ready
}

/** The core's flattened legal moves (`n, x, y, x, y, ...` per move) as cell lists. */
export function parseMoves(flat: ArrayLike<number>): Move[] {
  const moves: Move[] = []
  for (let i = 0; i < flat.length; ) {
    const n = flat[i]!
    const move: Move = []
    for (let k = 0; k < n; k++) move.push(cellOf(flat[i + 1 + 2 * k]!, flat[i + 2 + 2 * k]!))
    moves.push(move)
    i += 1 + 2 * n
  }
  return moves
}

export class CheckersEngine implements VersusEngine<CheckersPosition> {
  readonly players = ["Red", "Black"] as const
  readonly drawNote = `${MAX_MOVES_WITHOUT_CAPTURE} moves without a capture`
  readonly game: CheckersGame
  /** Piece ids, parallel to `game.cells()`. The core keeps its board in insertion order -- a moved piece is
   *  re-inserted last, a captured one removed -- so replaying that on the ids keeps each piece's identity. */
  private ids: number[]

  private constructor(game: CheckersGame) {
    this.game = game
    this.ids = this.freshIds()
  }

  static async create(): Promise<CheckersEngine> {
    await ensureCheckersReady()
    return new CheckersEngine(new CheckersGame(MAX_MOVES_WITHOUT_CAPTURE))
  }

  private freshIds() {
    return Array.from({ length: this.game.cells().length / 3 }, (_, i) => i)
  }

  reset() {
    this.game.reset()
    this.ids = this.freshIds()
  }
  legalMoves() {
    return parseMoves(this.game.legalMoves())
  }
  currentPlayer() {
    return this.game.currentPlayer
  }
  play(index: number) {
    const move = this.legalMoves()[index]
    const flat = this.game.cells()
    const at = (cell: number) => {
      for (let i = 0; i < flat.length; i += 3) if (cellOf(flat[i]!, flat[i + 1]!) === cell) return i / 3
      return -1
    }
    const mover = move ? at(move[0]!) : -1
    const captured = new Set<number>()
    for (let k = 1; move && k < move.length; k++) if (isJump(move[k - 1]!, move[k]!)) captured.add(at(jumpedSquare(move[k - 1]!, move[k]!)))
    this.game.step(index)
    if (mover >= 0) {
      const survivors = this.ids.filter((_, i) => i !== mover && !captured.has(i))
      this.ids = [...survivors, this.ids[mover]!]
    }
    // Self-heal if the bookkeeping ever disagrees with the core (it must not; but a wrong id only costs an animation).
    if (this.ids.length !== this.game.cells().length / 3) this.ids = this.freshIds()
  }
  isDone() {
    return this.game.done
  }
  winner() {
    return this.game.done && this.game.winner >= 0 ? this.game.winner : null
  }
  position(): CheckersPosition {
    const flat = this.game.cells()
    const cells: CheckersPiece[] = []
    for (let i = 0; i < flat.length; i += 3) cells.push({ id: this.ids[i / 3]!, x: flat[i]!, y: flat[i + 1]!, label: PIECES[flat[i + 2]!]! })
    return { width: BOARD_SIZE, height: BOARD_SIZE, cells }
  }
  notate(move: Move) {
    return move.map(squareName).join(isJump(move[0]!, move[1]!) ? "×" : "–")
  }
  spawn() {
    return new CheckersEngine(new CheckersGame(MAX_MOVES_WITHOUT_CAPTURE))
  }
  dispose() {
    this.game.free()
  }
}

/** A trained position evaluator, as the game core takes it: a layered network (an `evolve.WeightVector`'s flat
 *  weights and layer sizes) or a NEAT genome compiled to its evaluation plan (`NeatGenome.graph_encoding()`). */
export type CheckersBrain =
  | { kind: "layered"; weights: readonly number[]; layerSizes: readonly number[] }
  // `encoding` is what the core evaluates; `genome` is the same network as a graph, for the diagram to draw.
  | { kind: "graph"; encoding: readonly number[]; genome: Genome }

/** A brain as `/runs/{id}/artifacts/{ref}/brain` serves it (`evolve.networks.compiled` + the genome for a graph). */
export type BrainPayload =
  | { kind: "layered"; weights: number[]; layer_sizes: number[] }
  | { kind: "graph"; encoding: number[]; genome: GenomeJson }

export function brainFromApi(payload: BrainPayload): CheckersBrain {
  return payload.kind === "layered"
    ? { kind: "layered", weights: payload.weights, layerSizes: payload.layer_sizes }
    : { kind: "graph", encoding: payload.encoding, genome: genomeFromJson(payload.genome) }
}

export interface WasmStrategyOptions {
  /** Whatever the caller keys seats by (a leaderboard entrant id, or the kind itself). */
  id: string
  /** The core's name for it: `random`, `first-legal`, `material-N`, `evaluator`. */
  kind: string
  label: string
  description: string
  /** Only for `evaluator`. */
  brain?: CheckersBrain
  /** Alpha-beta plies an evaluator searches (default 1 = one ply). */
  depth?: number
}

/** Split the core's flat activations (every layer concatenated) back into layers. */
function splitLayers(flat: ArrayLike<number>, layerSizes: readonly number[]): number[][] {
  const layers: number[][] = []
  let at = 0
  for (const size of layerSizes) {
    layers.push(Array.from({ length: size }, (_, i) => flat[at + i]!))
    at += size
  }
  return layers
}

function wasmBot(engine: unknown, { kind, brain, depth }: WasmStrategyOptions, seed: number): Bot {
  const game = (engine as CheckersEngine).game
  const strategy =
    brain?.kind === "layered"
      ? new CheckersStrategy(kind, seed, Float64Array.from(brain.weights), Uint32Array.from(brain.layerSizes), depth)
      : brain?.kind === "graph"
        ? new CheckersStrategy(kind, seed, undefined, undefined, depth, Float64Array.from(brain.encoding))
        : new CheckersStrategy(kind, seed)
  const network: BotNetwork | undefined =
    brain?.kind === "layered"
      ? { kind: "layered", weights: brain.weights, layerSizes: brain.layerSizes, inputLabels: INPUT_LABELS, outputLabels: ["value"] }
      : brain?.kind === "graph"
        ? { kind: "graph", genome: brain.genome, inputLabels: INPUT_LABELS, outputLabels: ["value"] }
        : undefined
  return {
    pick: () => strategy.pick(game),
    scores: () => {
      const scores = strategy.scores(game)
      return scores ? Array.from(scores) : null
    },
    network,
    // What the diagram lights up: the network's values on the position legal move `index` leaves behind. A layered
    // network's come from the core itself; for a graph, the same position (the core's `simulate`) is run through
    // the visualization-only NEAT port the Snake champion diagram uses (utils/neat.ts, checked against Python).
    activations: network
      ? (index) => {
          if (network.kind === "layered") {
            const flat = strategy.activations(game, index)
            return flat ? splitLayers(flat, network.layerSizes) : null
          }
          return neatActivations(network.genome, Array.from(game.simulate(index)))
        }
      : undefined,
    free: () => strategy.free(),
  }
}

/** A Rust player (rust/core/src/checkers_strategies.rs) as a `VersusStrategy`. */
export function wasmStrategy(options: WasmStrategyOptions): VersusStrategy<CheckersPosition> {
  const { id, label, description } = options
  return { id, label, description, create: (engine, seed) => wasmBot(engine, options, seed) }
}

const builtin = (kind: string, label: string, description: string) => wasmStrategy({ id: kind, kind, label, description })

/** The static players -- the same ones `games.checkers_strategies` (Python) exposes; jobs/evaluate_versus.py
 *  ranks them as the leaderboard's baselines. Used directly when there is no leaderboard to draw entrants from. */
export const STATIC_STRATEGIES: VersusStrategy<CheckersPosition>[] = [
  builtin("random", "Random", "A uniformly random legal move."),
  builtin("first-legal", "First legal", "Always the first legal move: dumb but perfectly consistent."),
  builtin(
    "material-1",
    "Material, 1-ply",
    "The move that leaves the best material balance. Barely beats random: mandatory captures leave it almost nothing to choose between.",
  ),
  builtin(
    "material-2",
    "Material, 2-ply",
    "Assumes the opponent answers with their best reply, and plays the move that holds up: a tiny minimax.",
  ),
  builtin("material-3", "Material, 3-ply", "The same material search, three plies deep with alpha-beta."),
  builtin("material-4", "Material, 4-ply", "The same material search, four plies deep with alpha-beta: the fair opponent for a trained evaluator that searches as deep."),
]

/** A trained position evaluator played by the Rust `evaluator` strategy (one ply ahead by default, or
 *  `depth` plies of alpha-beta with the network scoring the leaves). */
export function evolvedStrategy(
  id: string,
  label: string,
  description: string,
  weights: readonly number[],
  layerSizes: readonly number[],
  depth = 1,
): VersusStrategy<CheckersPosition> {
  return wasmStrategy({ id, kind: "evaluator", label, description, brain: { kind: "layered", weights, layerSizes }, depth })
}
