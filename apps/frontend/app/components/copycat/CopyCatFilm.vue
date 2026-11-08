<script setup lang="ts">
import type { CopyCatImageScores, CopyCatTargetDetail } from "~/utils/copycat"
import type { ChartSeries } from "~/types/chart"

// Copy Cat's stage: the entrant's SVG for this image beside what it was given and the clean SVG it is scored against,
// with the image's scores and its shapes as layers pulled apart -- and, as a timelapse (docs/design/0023), how it got
// there: each change it kept, played back, paused, scrubbed or stepped, with one score for every entrant under it, ending
// on the SVG the leaderboard scored. The entrant really runs here, in WebAssembly, so what plays is what it does: an
// entrant with no process to record shows only its SVG, and gets no invented animation. While the pipeline runs, the
// stage shows its SVG so far, and how long it has left.
const props = defineProps<{
  entrantId: string | null
  entrantLabel: string
  detail: CopyCatTargetDetail | null
  result: { svg: string; scores: CopyCatImageScores } | null
  givenNote?: string
  /** For your own image: what its SVG downloads as. */
  downloadName?: string
}>()

const recipe = computed(() => filmRecipe(props.entrantId))
const subject = computed(() => {
  const d = props.detail
  return d && recipe.value ? { png: d.png, seed: d.seed, clean: d.clean, recipe: recipe.value } : null
})
const { film, scores, failed, live } = useCopyCatFilm(subject)

const frames = computed(() => film.value?.frames ?? [])
const times = computed(() => filmTimes(frames.value))
const duration = computed(() => times.value[times.value.length - 1] ?? 0)

const time = ref(0)
const paused = ref(false)
const speed = ref(1)
const SPEED_CHOICES = [0.25, 0.5, 1, 2, 4] as const
const current = computed(() => frameAt(times.value, time.value))
const ended = computed(() => duration.value > 0 && time.value >= duration.value)

/** A run the reader watched for longer than this (ms) has been seen being built: its film waits at the end. */
const WATCHED_LIVE = 1500
// a new film starts from the top, playing unless the reader asked for less motion -- or just watched it being made
watch(film, (next) => {
  if (next && (live.value?.elapsed ?? 0) > WATCHED_LIVE) {
    time.value = filmTimes(next.frames).at(-1) ?? 0
    paused.value = true
    return
  }
  time.value = 0
  paused.value = !next || (import.meta.client && window.matchMedia("(prefers-reduced-motion: reduce)").matches)
})

let raf = 0
let last = 0
function tick(now: number) {
  if (!paused.value && !ended.value) time.value = Math.min(duration.value, time.value + ((now - last) / 1000) * speed.value)
  last = now
  raf = requestAnimationFrame(tick)
}
onMounted(() => {
  last = performance.now()
  raf = requestAnimationFrame(tick)
})
onBeforeUnmount(() => cancelAnimationFrame(raf))

function setPaused(next: boolean) {
  if (!next && ended.value) time.value = 0
  paused.value = next
}
const replay = () => {
  time.value = 0
  paused.value = false
}
function seekFrame(i: number) {
  paused.value = true
  const clamped = Math.max(0, Math.min(frames.value.length - 1, i))
  time.value = times.value[clamped] ?? 0
}
function onScrub(event: Event) {
  paused.value = true
  time.value = Number((event.target as HTMLInputElement).value)
}
const toEnd = () => {
  paused.value = true
  time.value = duration.value
}

const frame = computed(() => frames.value[current.value] ?? null)
const elements = computed(() => elementsAt(frames.value, current.value))
const inner = computed(() => elements.value.join(""))

// a run in progress, before its film is ready
const running = computed(() => (film.value ? null : live.value))
const leftNote = computed(() => {
  const left = running.value?.left
  if (left === null || left === undefined) return "estimating time left…"
  return left < 1000 ? "under a second left" : `about ${Math.round(left / 1000)} s left`
})

// the shapes on stage, each one its own SVG, for the layers pulled apart: the frame's, the run's so far, or the scored SVG's
const SHAPE = /<(?:rect|circle|ellipse|polygon|path)\b[^>]*\/>/g
const layers = computed(() => {
  const d = props.detail
  if (!d) return []
  const shown = film.value ? elements.value : (running.value?.elements ?? props.result?.svg ?? "").match(SHAPE) ?? []
  return shown.map((el) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${d.width} ${d.height}">${el}</svg>`)
})

function labelImage(labels: Uint16Array | undefined) {
  const d = props.detail
  if (!labels?.length || !d || !film.value) return null
  return { data: labelPixels(labels, film.value.palette), width: d.width, height: d.height }
}
const clusteredImage = computed(() => labelImage(film.value?.clustered))
const labelsImage = computed(() => labelImage(film.value?.labels))
const stageCanvas = ref<HTMLCanvasElement | null>(null)
const pixels = computed(() => (frame.value?.view === "clustered" ? clusteredImage.value : frame.value?.view === "labels" ? labelsImage.value : null))
watchEffect(() => {
  const el = stageCanvas.value
  const p = pixels.value
  if (!el || !p) return
  el.width = p.width
  el.height = p.height
  el.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(p.data), p.width, p.height), 0, 0)
})

// the timeline, in the entrant's own stages
const spans = computed(() => {
  const out: { stage: string; start: number; share: number; first: number }[] = []
  frames.value.forEach((f, i) => {
    const t = times.value[i]!
    const open = out[out.length - 1]
    if (open?.stage === f.stage) return
    out.push({ stage: f.stage, start: t, share: 0, first: i })
  })
  out.forEach((span, i) => {
    span.share = ((out[i + 1]?.start ?? duration.value) - span.start) / (duration.value || 1)
  })
  return out
})
const activeSpan = computed(() => {
  let found = 0
  spans.value.forEach((span, i) => {
    if (span.first <= current.value) found = i
  })
  return found
})

// the charts: clean fidelity for every entrant, its own objective beside it, on the frames that stayed on its path
const scored = computed(() => [...scores.value.entries()].sort((a, b) => a[0] - b[0]))
const fidelityNow = computed(() => {
  let best: number | null = null
  for (const [i, value] of scored.value) {
    if (i > current.value) break
    best = value
  }
  return best
})
const inDb = computed(() => film.value?.objectiveLabel.includes("dB") ?? false)
const x = computed(() => scored.value.map(([i]) => i))
const fidelitySeries = computed<ChartSeries[]>(() => {
  const series: ChartSeries[] = [{ key: "fidelity", label: "clean fidelity (dB)", color: palette.queen400, values: scored.value.map(([, v]) => v), width: 2 }]
  if (inDb.value) {
    series.push({ key: "input", label: film.value!.objectiveLabel, color: palette.signal400, values: scored.value.map(([i]) => frames.value[i]!.objective), width: 2 })
  }
  return series
})
const shapesAt = (i: number) => elementsAt(frames.value, i).length
const secondSeries = computed<ChartSeries[]>(() =>
  inDb.value
    ? [{ key: "shapes", label: "shapes", color: palette.gold400, values: scored.value.map(([i]) => shapesAt(i)), width: 2 }]
    : [{ key: "objective", label: film.value?.objectiveLabel ?? "", color: palette.signal400, values: scored.value.map(([i]) => frames.value[i]!.objective), width: 2 }],
)
const allScored = computed(() => film.value !== null && scores.value.size === samplesOf(frames.value).length && scores.value.size > 0)

const stats = computed(() => {
  if (!frame.value) return []
  const objective = frame.value.objective
  return [
    { label: "Frame", value: `${current.value + 1} / ${frames.value.length}` },
    { label: "Shapes", value: elements.value.length },
    { label: "Clean fidelity", value: fidelityNow.value === null ? "--" : `${fidelityNow.value.toFixed(1)} dB`, tone: "queen" as const },
    { label: inDb.value ? "Fit to the input" : "Description", value: !Number.isFinite(objective) ? "--" : inDb.value ? `${objective.toFixed(1)} dB` : `${Math.round(objective).toLocaleString()} bits` },
  ]
})

// the rule the film is held to: it ends on exactly what the leaderboard scored
const lastScore = computed(() => {
  const last = scores.value.get(frames.value.length - 1)
  return last === undefined ? null : last
})
const sameAsScored = computed(() => {
  if (!film.value || !props.result) return null
  if (film.value.svg !== props.result.svg || !film.value.matchesOutput) return false
  if (allScored.value && lastScore.value !== null) return Math.abs(lastScore.value - props.result.scores.fidelity) < 0.001
  return true
})
const yours = computed(() => props.detail?.id === OWN_IMAGE)
function download() {
  if (!film.value) return
  const url = URL.createObjectURL(new Blob([film.value.svg], { type: "image/svg+xml" }))
  const a = document.createElement("a")
  a.href = url
  a.download = `${props.downloadName ?? "traced"}.svg`
  a.click()
  URL.revokeObjectURL(url)
}
// a phone's width fits two readouts a row; three or four would cut every value short
const narrow = ref(false)
let narrowQuery: MediaQueryList | null = null
const onNarrow = () => (narrow.value = narrowQuery?.matches ?? false)
onMounted(() => {
  narrowQuery = window.matchMedia("(max-width: 639px)")
  onNarrow()
  narrowQuery.addEventListener("change", onNarrow)
})
onBeforeUnmount(() => narrowQuery?.removeEventListener("change", onNarrow))
const KB = (n: number) => `${(n / 1024).toFixed(0)} KB`
const filmBytes = computed(() => frames.value.reduce((sum, f) => sum + (f.element?.length ?? 0) + f.note.length + f.stage.length + 16, 0))
const hasRun = computed(() => recipe.value !== null)

// before a film is ready (or for an entrant with none), the stage shows the SVG the leaderboard scored
const scoredUrl = computed(() => (props.result ? svgDataUrl(props.result.svg) : null))
const countOf = (svg: string) => svg.match(/<(rect|circle|ellipse|polygon|path)\b/g)?.length ?? 0
const cleanShapes = computed(() => (props.detail ? countOf(props.detail.clean) : null))
const cleanColours = computed(() => (props.detail ? new Set(props.detail.clean.match(/fill="#[0-9a-f]{6}"/g) ?? []).size : null))
const scoredStats = computed(() => {
  const s = props.result?.scores
  if (!s) return []
  return [
    { label: "Clean fidelity", value: `${s.fidelity.toFixed(1)} dB`, tone: "queen" as const },
    { label: "Alpha IoU", value: s.alpha_iou.toFixed(2) },
    { label: "Layer recovery", value: s.layer_recovery.toFixed(2) },
    { label: "Shapes", value: `${s.shapes} (clean ${cleanShapes.value ?? "?"})` },
    { label: "Colours", value: `${s.palette} (clean ${cleanColours.value ?? "?"})` },
    { label: "Size", value: `${(s.bytes / 1024).toFixed(1)} KB` },
  ]
})
</script>

<template>
  <!-- the stage's own panel (GamePage) already names the entrant, so this is its body, not a panel of its own -->
  <div class="space-y-4">
      <div class="grid gap-4 md:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        <figure class="min-w-0">
          <div class="checker relative aspect-square overflow-hidden rounded-[8px] border border-line" data-testid="film-stage">
            <template v-if="film">
              <img v-if="frame?.view === 'input'" :src="detail?.png" alt="The image it was given" class="pixelated size-full" />
              <canvas v-else-if="pixels" ref="stageCanvas" class="pixelated size-full" />
              <!-- eslint-disable-next-line vue/no-v-html -- our own WASM's numbers, never text from an image or a person -->
              <svg v-else xmlns="http://www.w3.org/2000/svg" :viewBox="`0 0 ${detail?.width} ${detail?.height}`" class="size-full" v-html="inner" />
            </template>
            <!-- eslint-disable-next-line vue/no-v-html -- our own WASM's numbers, never text from an image or a person -->
            <svg v-else-if="running" xmlns="http://www.w3.org/2000/svg" :viewBox="`0 0 ${detail?.width} ${detail?.height}`" class="size-full" v-html="running.elements" />
            <img v-else-if="scoredUrl" :src="scoredUrl" :alt="`${entrantLabel}'s SVG`" class="size-full object-contain" draggable="false" />
            <UiEmpty v-else class="absolute inset-0" compact>{{ !entrantId || !detail ? "Loading…" : yours ? "This entrant can't run here" : "No SVG recorded for this image" }}</UiEmpty>
          </div>
          <figcaption class="mt-1.5 flex items-baseline justify-between gap-3 text-xs">
            <template v-if="film">
              <span class="shrink-0 font-medium text-fg">{{ frame?.stage }}</span>
              <span :title="frame?.note" class="min-w-0 truncate font-mono text-[11px] text-fg-subtle">{{ frame?.note }}</span>
            </template>
            <template v-else-if="running">
              <span class="shrink-0 font-medium text-fg">{{ running.stage || "Reading the image" }}</span>
              <span :title="running.note" class="min-w-0 truncate font-mono text-[11px] text-fg-subtle">{{ running.note || "noise, background, palette, parts" }}</span>
            </template>
            <template v-else>
              <span class="shrink-0 font-medium text-fg">The SVG it wrote</span>
              <span v-if="hasRun && !failed" class="min-w-0 truncate text-[11px] text-fg-subtle">running the timelapse…</span>
            </template>
          </figcaption>
          <div v-if="running" class="mt-2 space-y-1" data-testid="film-progress">
            <div
              class="h-1.5 overflow-hidden rounded-full bg-line"
              role="progressbar"
              aria-label="Tracing"
              :aria-valuenow="Math.round(running.done * 100)"
              aria-valuemin="0"
              aria-valuemax="100"
            >
              <div class="h-full bg-queen-400 transition-[width] duration-150" :style="{ width: `${running.done * 100}%` }" />
            </div>
            <p class="flex justify-between font-mono text-[11px] text-fg-subtle">
              <span>{{ Math.round(running.done * 100) }}% · {{ (running.elapsed / 1000).toFixed(1) }} s</span>
              <span>{{ leftNote }}</span>
            </p>
          </div>
          <div class="mt-3">
            <slot name="picker" />
          </div>
        </figure>
        <div class="min-w-0 space-y-4">
          <div class="grid max-w-[540px] grid-cols-3 gap-3">
            <CopyCatImage label="Given" :src="detail?.png" :note="givenNote" />
            <CopyCatImage label="Clean, never seen" :svg="detail?.clean" :note="cleanShapes ? `${cleanShapes} shapes` : undefined">
              <template #empty>{{ yours ? "None: it's your image" : "Loading…" }}</template>
            </CopyCatImage>
            <CopyCatLayers :layers="layers" label="Its layers" />
          </div>
          <p v-if="detail?.source" class="text-[11px] text-fg-subtle">
            Clean SVG:
            <a :href="detail.source.url" class="underline hover:text-fg" target="_blank" rel="noopener">{{ detail.source.credit }}</a>
            ({{ detail.source.licence }}), rendered and corrupted here.
          </p>
          <div v-if="scoredStats.length">
            <div class="mb-1.5 flex flex-wrap items-center justify-between gap-2">
              <p class="label">Scored on this image</p>
              <UiBadge v-if="sameAsScored === true" tone="life" dot>the timelapse ends on this SVG</UiBadge>
              <UiBadge v-else-if="sameAsScored === false" tone="gold" dot title="The entrant run here wrote a different SVG from the one the leaderboard scored: the code has changed since it was scored">
                the timelapse ends on a different SVG
              </UiBadge>
            </div>
            <UiStats :items="scoredStats" :cols="narrow ? 2 : 3" />
          </div>
          <p v-if="result?.scores.refused" class="text-xs text-queen-300">Refused: {{ result.scores.refused }} -- scored as an empty SVG.</p>
          <div v-if="film">
            <p class="label mb-1.5">This frame</p>
            <UiStats :items="stats" :cols="narrow ? 2 : 4" />
          </div>
          <div v-if="film && yours" class="flex flex-wrap items-center gap-3">
            <button type="button" class="btn-primary btn-sm" @click="download">Download {{ downloadName ?? "traced" }}.svg</button>
            <span class="text-xs text-fg-subtle">Traced on this device; nothing was uploaded. No clean SVG, so no score.</span>
          </div>
          <p v-else-if="!hasRun && yours" class="text-xs text-fg-subtle">
            This entrant doesn't run in the browser. Pick the MDL tracer or a hill climb to trace your image.
          </p>
          <p v-else-if="!hasRun" class="text-xs text-fg-subtle">
            An outside tool that hands back only its result, so there is no process to play. Pick the MDL tracer or a hill climb to watch one at work.
          </p>
          <p v-else-if="failed" class="text-xs text-fg-subtle">The timelapse couldn't be run in this browser ({{ failed }}).</p>
        </div>
      </div>

      <template v-if="film">
      <div class="space-y-2">
        <input
          type="range"
          class="w-full accent-queen-400"
          aria-label="Position in the film"
          :min="0"
          :max="duration"
          step="any"
          :value="time"
          data-testid="film-scrub"
          @input="onScrub"
        />
        <div class="flex h-5 w-full gap-px overflow-hidden rounded-[4px]" role="group" aria-label="Stages">
          <button
            v-for="(span, i) in spans"
            :key="span.first"
            type="button"
            class="min-w-px transition-colors"
            :class="i === activeSpan ? 'bg-queen-400' : i < activeSpan ? 'bg-queen-400/35' : 'bg-line hover:bg-line-strong'"
            :style="{ flexGrow: Math.max(span.share, 0.001), flexBasis: 0 }"
            :title="span.stage"
            :aria-label="`Jump to ${span.stage}`"
            @click="seekFrame(span.first)"
          />
        </div>
      </div>

      <PlaybackControls
        :paused="paused || ended"
        :speed="speed"
        :speeds="SPEED_CHOICES"
        new-game-label="Replay"
        @update:paused="setPaused"
        @update:speed="(s) => (speed = s)"
        @new-game="replay"
      >
        <span class="order-first flex gap-1" role="group" aria-label="Step">
          <button class="btn-ghost btn-sm" aria-label="Previous frame" @click="seekFrame(current - 1)">‹</button>
          <button class="btn-ghost btn-sm" aria-label="Next frame" @click="seekFrame(current + 1)">›</button>
          <button class="btn-ghost btn-sm" aria-label="Last frame" @click="toEnd">⇥</button>
        </span>
      </PlaybackControls>

      <div v-if="x.length > 1" class="grid gap-4 md:grid-cols-2">
        <div>
          <UiLegend :series="fidelitySeries" />
          <LineChart :x="x" :series="fidelitySeries" x-label="frame" :height="160" :marker="current" clickable :format="(v) => v.toFixed(1)" @select="seekFrame" />
        </div>
        <div>
          <UiLegend :series="secondSeries" />
          <LineChart :x="x" :series="secondSeries" x-label="frame" :height="160" :marker="current" clickable :format="(v) => Math.round(v).toLocaleString()" @select="seekFrame" />
        </div>
      </div>
      <p class="text-xs text-fg-subtle" data-testid="film-size">
        Run just now in your browser: {{ Math.round(film.ms) }} ms, {{ frames.length }} frames, {{ KB(filmBytes) }}<template v-if="!allScored && !yours">, still scoring frames</template>. Steps are thinned to
        a few per shape. Clean fidelity is what the leaderboard ranks; the entrant never sees the clean SVG.
      </p>
      </template>
  </div>
</template>

<style scoped>
.checker {
  background-color: var(--color-raised);
  background-image:
    linear-gradient(45deg, var(--color-line) 25%, transparent 25%, transparent 75%, var(--color-line) 75%),
    linear-gradient(45deg, var(--color-line) 25%, transparent 25%, transparent 75%, var(--color-line) 75%);
  background-size: 16px 16px;
  background-position:
    0 0,
    8px 8px;
}
.pixelated {
  image-rendering: pixelated;
}
</style>
