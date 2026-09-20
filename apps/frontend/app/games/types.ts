import type { Component, Ref } from "vue"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { DeviceFit } from "~/types/modelpack"

// How a game plugs into the shared game page (docs/design/0007). Every game page has the same skeleton --
// header and Watch/Play toggle, a live stage, a ranked side leaderboard, then the full leaderboard with
// its measurements and representations -- and `GamePage.vue` renders it once, for any game. A game is a
// small `GameModule` saying what differs: what its score means and how to print it, which extra
// leaderboard columns it has, its stage components, and how to describe itself. Adding a game means
// writing a module (app/games/<slug>.ts), registering it (app/games/registry.ts) and, for a two-player
// game, giving it a `VersusEngine` -- not another page.

/** What a game's score is and how it's shown. */
export interface ScoreSpec {
  /** Column/heading name: "Held-out score", "Points per game". */
  label: string
  format(value: number): string
  /** The same, tighter, for the side list. */
  compact(value: number): string
  /** Bars are scaled to at least this: a score with a natural ceiling (points per game: 1) fills its bar. */
  scaleMin: number
}

/** A leaderboard cell for a game-specific column; null leaves it empty. */
export interface CellSpec {
  text: string
  sub?: string
  tone?: "warn" | "muted"
}

/** A game-specific leaderboard column, beside the generic ones (rank, entrant, representation, score,
 *  inference, parameters, training). */
export interface LeaderboardColumn {
  id: string
  header: string
  title?: string
  cell(record: EvaluationRecord): CellSpec | null
}

/** What *this device* can run of a game's entrants. Snake's champions are ONNX model packages that need a
 *  capable backend; a game whose entrants run anywhere uses `noDevice`. */
export interface GameDevice {
  load(): void
  /** Per entrant: can this device run it, and why not. Entrants absent from the map are assumed to run. */
  runsHere: Ref<Record<string, DeviceFit>>
  /** One line for the leaderboard footer ("WebGPU · WASM threads"), or null. */
  summary: Ref<string | null>
  /** Whether the "only what runs here" filter makes sense yet. */
  filterable: Ref<boolean>
  /** Whatever the game's own stages need from it (Snake: the model catalog). Opaque to the page. */
  context: unknown
}

/** The props every stage component receives. A stage emits `score` (the human's score, for the side
 *  leaderboard's "You" row) and `exit` (leave play mode) if it wants to. */
export interface StageProps {
  mode: "watch" | "play"
  /** The leaderboard entrant on stage (watched, or the opponent in play mode); null before there are any. */
  entry: EvaluationRecord | null
  entries: EvaluationRecord[]
  device: GameDevice
}

export interface GameModule {
  slug: string
  score: ScoreSpec
  /** Game-specific leaderboard columns, between the score and the cost columns. */
  columns: LeaderboardColumn[]
  /** What opens on the stage: the best entrant that *learned* (Snake: watch something trained), or simply
   *  the top of the leaderboard (Checkers: the strongest player, baselines included). */
  defaultEntrant: "best-trained" | "top"
  /** Clicking an entrant while playing: Snake's Play is solo, so it switches to watching that entrant; a
   *  versus game's Play *is* against an entrant, so clicking picks your opponent and stays in Play. */
  selectInPlay: "watch" | "stay"
  copy: {
    watch: string
    play: string
    /** What a score means, for the footer under the side leaderboard. */
    scoreNote(info: { episodes?: number; protocol: string | null }): string
    /** The full leaderboard's intro. */
    leaderboardIntro(info: { episodes?: number; seeds?: [number, number] }): string
  }
  /** The stage for Watch mode and for Play mode (they may be the same component, told apart by `mode`). */
  Watch: Component
  Play: Component
  /** Optional device probing (see GameDevice). */
  device?(slug: string, entries: Ref<EvaluationRecord[]>): GameDevice
  /** The visitor's stored personal result, for the "You" row before they play. */
  initialHuman?(slug: string): { score: number; label: string; live: boolean } | null
  /** Optional page sections. */
  sections: { pareto: boolean; headToHead: boolean; representations: boolean }
}
