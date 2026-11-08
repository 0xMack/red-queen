// An entrant at work (docs/design/0023): runs it here, in WebAssembly, with the recorder on, and posts its frames, then
// each frame's clean fidelity as they're scored. A worker, because a film of an icon can take seconds, and scoring renders
// every sampled frame.
import init, { CopyCatFilm } from "~/wasm/games/games.js"
import gamesWasmUrl from "~/wasm/games/games_bg.wasm?url"
import type { FilmFrame, FilmMessage, FilmRequest } from "~/utils/copycatFilm"
import { samplesOf } from "~/utils/copycatFilm"

const SCORES_PER_POST = 6
/** The pipeline's progress, posted at most this often (ms): a part can take a few milliseconds. */
const PROGRESS_EVERY = 100

let ready: Promise<unknown> | null = null
let latest = 0

function record(request: FilmRequest, started: number): CopyCatFilm {
  const { rgba, width, height, recipe } = request
  if (recipe.kind === "pipeline") {
    // the run is one synchronous call, so it tells us how far it is as it goes; we pass that on, with the SVG so far
    let last = -Infinity
    return CopyCatFilm.pipeline_live(rgba, width, height, (done: number, stage: string, note: string, elements: string) => {
      const now = performance.now()
      if (done < 1 && now - last < PROGRESS_EVERY) return
      last = now
      self.postMessage({ type: "progress", id: request.id, done, stage, note, elements, elapsed: now - started } satisfies FilmMessage)
    })
  }
  return CopyCatFilm.hill_climb(rgba, width, height, recipe.shapes, recipe.candidates, recipe.patience, BigInt(request.seed), recipe.dropBottom)
}

self.onmessage = async (event: MessageEvent<FilmRequest>) => {
  const request = event.data
  latest = request.id
  const post = (message: FilmMessage) => self.postMessage(message)
  try {
    ready ??= init({ module_or_path: gamesWasmUrl })
    await ready
    if (request.id !== latest) return
    const started = performance.now()
    const film = record(request, started)
    const ms = performance.now() - started
    try {
      const frames: FilmFrame[] = Array.from({ length: film.frames }, (_, i) => ({
        stage: film.stage(i),
        note: film.note(i),
        kind: film.kind(i) as FilmFrame["kind"],
        view: film.view(i) as FilmFrame["view"],
        index: film.patch_index(i),
        element: film.element(i) ?? null,
        objective: film.objective(i),
      }))
      const flat = film.palette()
      const palette: [number, number, number][] = []
      for (let i = 0; i + 2 < flat.length; i += 3) palette.push([flat[i]!, flat[i + 1]!, flat[i + 2]!])
      post({
        type: "film",
        id: request.id,
        frames,
        objectiveLabel: film.objective_label,
        svg: film.svg(),
        matchesOutput: film.matches_output(),
        palette,
        clustered: film.clustered(),
        labels: film.labels(),
        ms,
      })

      // your own image has no clean SVG, so nothing to score against
      if (!request.clean) return
      film.set_clean(request.clean)
      const samples = samplesOf(frames)
      let batch: [number, number][] = []
      for (const i of samples) {
        batch.push([i, film.fidelity_at(i)])
        if (batch.length === SCORES_PER_POST) {
          post({ type: "scores", id: request.id, scores: batch })
          batch = []
          // lets a newer request in: a film of an image no one is looking at isn't worth finishing
          await new Promise((resolve) => setTimeout(resolve))
          if (request.id !== latest) return
        }
      }
      post({ type: "scores", id: request.id, scores: batch })
    } finally {
      film.free()
    }
  } catch (e) {
    post({ type: "error", id: request.id, message: e instanceof Error ? e.message : String(e) })
  }
}
