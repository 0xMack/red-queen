// What a film of a Copy Cat entrant is, between the worker that records it and the player that shows it (docs/design/0023).

/** How to run an entrant with the recorder on. `null` for an entrant with no process to record (VTracer). */
export type FilmRecipe =
  | { kind: "pipeline" }
  | { kind: "hillclimb"; shapes: number; candidates: number; patience: number; dropBottom: boolean }

// The evaluator's settings (jobs/evaluator/.../copycat_clean.py): the leaderboard's entrants, run again here. The badge
// on the player shows if they ever stop agreeing.
const RECIPES: Record<string, FilmRecipe> = {
  "baseline:structured": { kind: "pipeline" },
  "baseline:hillclimb-16": { kind: "hillclimb", shapes: 16, candidates: 100, patience: 100, dropBottom: false },
  "baseline:hillclimb-64": { kind: "hillclimb", shapes: 64, candidates: 100, patience: 100, dropBottom: false },
  "baseline:hillclimb-64-nobg": { kind: "hillclimb", shapes: 64, candidates: 100, patience: 100, dropBottom: true },
}
export const filmRecipe = (entrantId: string | null | undefined): FilmRecipe | null => (entrantId ? (RECIPES[entrantId] ?? null) : null)

export interface FilmRequest {
  id: number
  rgba: Uint8Array
  width: number
  height: number
  seed: number
  clean: string
  recipe: FilmRecipe
}

export interface FilmFrame {
  stage: string
  note: string
  /** How much it matters: `try` is a candidate, off the path the charts follow. */
  kind: "stage" | "try" | "step" | "lock"
  view: "input" | "clustered" | "labels" | "vector"
  /** The shape it changes, or -1. */
  index: number
  /** That shape's new element; null with an index removes it. */
  element: string | null
  /** The entrant's own objective, NaN if it has none. */
  objective: number
}

export type FilmMessage =
  | {
      type: "film"
      id: number
      frames: FilmFrame[]
      objectiveLabel: string
      svg: string
      matchesOutput: boolean
      palette: [number, number, number][]
      clustered: Uint16Array
      labels: Uint16Array
      ms: number
    }
  | { type: "progress"; id: number; done: number; stage: string; note: string; elements: string; elapsed: number }
  | { type: "scores"; id: number; scores: [number, number][] }
  | { type: "error"; id: number; message: string }

/** Frames scored and charted: up to this many on the kept path, with the last always among them. */
const MAX_SAMPLES = 90

export function samplesOf(frames: readonly FilmFrame[]): number[] {
  const kept = frames.flatMap((f, i) => (f.kind === "try" ? [] : [i]))
  if (kept.length <= MAX_SAMPLES) return kept
  const out = Array.from({ length: MAX_SAMPLES }, (_, i) => kept[Math.floor((i * kept.length) / MAX_SAMPLES)]!)
  out[MAX_SAMPLES - 1] = kept[kept.length - 1]!
  return out
}

/** The shapes in place at frame `i`, bottom first: frames are patches, replayed from the start. */
export function elementsAt(frames: readonly FilmFrame[], i: number): string[] {
  const elements: string[] = []
  for (let k = 0; k <= i && k < frames.length; k++) {
    const { index, element } = frames[k]!
    if (index < 0) continue
    if (element === null) elements.splice(index, 1)
    else elements[index] = element
  }
  return elements
}

// How long a frame holds, relative to the others: a stage is read, a shape's final form lands, and a step goes by.
const WEIGHT: Record<FilmFrame["kind"], number> = { stage: 28, lock: 7, try: 5, step: 1 }

/** How long a film runs at 1×, whatever its frame count. */
export const FILM_SECONDS = 14

/** Per frame, when it starts, in seconds, and at the end the film's length. */
export function filmTimes(frames: readonly FilmFrame[]): number[] {
  const total = frames.reduce((sum, f) => sum + WEIGHT[f.kind], 0) || 1
  const times = [0]
  for (const f of frames) times.push(times[times.length - 1]! + (WEIGHT[f.kind] / total) * FILM_SECONDS)
  return times
}

/** The frame showing at `time`. */
export function frameAt(times: readonly number[], time: number): number {
  let low = 0
  let high = times.length - 2
  while (low < high) {
    const middle = Math.ceil((low + high) / 2)
    if (times[middle]! <= time) low = middle
    else high = middle - 1
  }
  return Math.max(0, low)
}
