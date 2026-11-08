import type { FilmFrame, FilmMessage, FilmRecipe, FilmRequest } from "~/utils/copycatFilm"

export interface CopyCatFilmData {
  frames: FilmFrame[]
  objectiveLabel: string
  svg: string
  matchesOutput: boolean
  palette: [number, number, number][]
  clustered: Uint16Array
  labels: Uint16Array
  ms: number
}

/** A run in progress: how far it is, where, the SVG so far (its elements), and about how long it has left. */
export interface CopyCatLive {
  done: number
  stage: string
  note: string
  elements: string
  elapsed: number
  /** ms, null until there is enough of the run to go on */
  left: number | null
}

/** One image to film. */
export interface FilmSubject {
  png: string
  seed: number
  clean: string
  recipe: FilmRecipe
}

// Records films in a worker, for whichever image and entrant is on stage: a new subject replaces the one in progress.
export function useCopyCatFilm(subject: Ref<FilmSubject | null>) {
  const film = shallowRef<CopyCatFilmData | null>(null)
  /** Frame -> clean fidelity, as they're scored. */
  const scores = shallowRef<Map<number, number>>(new Map())
  const failed = ref<string | null>(null)
  const live = shallowRef<CopyCatLive | null>(null)

  let worker: Worker | null = null
  let requested = 0

  function onMessage(event: MessageEvent<FilmMessage>) {
    const message = event.data
    if (message.id !== requested) return
    if (message.type === "progress") {
      const { done, stage, note, elements, elapsed } = message
      // time per unit of progress so far, for the rest: it is measured in pixels fitted, which the fits' time follows
      const left = done >= 0.05 && elapsed > 250 ? (elapsed * (1 - done)) / done : null
      live.value = { done, stage, note, elements, elapsed, left }
    } else if (message.type === "film") {
      const { type: _type, id: _id, ...data } = message
      film.value = data
    } else if (message.type === "scores") {
      const next = new Map(scores.value)
      for (const [i, score] of message.scores) next.set(i, score)
      scores.value = next
    } else {
      failed.value = message.message
    }
  }

  watch(
    subject,
    async (next) => {
      const id = ++requested
      film.value = null
      live.value = null
      scores.value = new Map()
      failed.value = null
      if (!next || !import.meta.client) return
      try {
        const pixels = await dataUrlPixels(next.png)
        if (id !== requested) return
        worker ??= new Worker(new URL("~/workers/copycatFilm.worker.ts", import.meta.url), { type: "module" })
        worker.onmessage = onMessage
        const request: FilmRequest = {
          id,
          rgba: new Uint8Array(pixels.data.buffer.slice(0)),
          width: pixels.width,
          height: pixels.height,
          seed: next.seed,
          clean: next.clean,
          recipe: toRaw(next.recipe),
        }
        worker.postMessage(request, [request.rgba.buffer])
      } catch (e) {
        failed.value = e instanceof Error ? e.message : String(e)
      }
    },
    { immediate: true },
  )

  onScopeDispose(() => {
    worker?.terminate()
    worker = null
  })

  return { film, scores, failed, live }
}
