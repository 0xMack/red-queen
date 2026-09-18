<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { formatFitness } from "~/utils/format"

// A small hand-rolled SVG line chart (still no charting-library dependency -- doc 0005): axes with
// "nice" ticks, an optional shaded band (e.g. worst..best fitness), a hover crosshair + tooltip,
// and an optional marker/click-to-select on the x axis (the run page pins a generation this way).
// Measures its own width with a ResizeObserver so text stays crisp at any size instead of being
// scaled by a viewBox.

const props = withDefaults(
  defineProps<{
    x: number[]
    series: ChartSeries[]
    band?: { lower: number[]; upper: number[]; color: string } | null
    height?: number
    marker?: number | null
    xLabel?: string
    clickable?: boolean
    format?: (v: number) => string
  }>(),
  { band: null, height: 280, marker: null, xLabel: "generation", clickable: false, format: (v: number) => formatFitness(v, 3) },
)

const emit = defineEmits<{ select: [x: number] }>()

const container = ref<HTMLElement | null>(null)
const width = ref(640)
let observer: ResizeObserver | null = null
onMounted(() => {
  observer = new ResizeObserver((entries) => {
    const w = entries[0]?.contentRect.width
    if (w) width.value = Math.max(240, Math.round(w))
  })
  if (container.value) observer.observe(container.value)
})
onUnmounted(() => observer?.disconnect())

const pad = { top: 16, right: 16, bottom: 30, left: 52 }

function niceTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min]
  const span = max - min
  const step0 = span / count
  const mag = 10 ** Math.floor(Math.log10(step0))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? 10 * mag
  const ticks: number[] = []
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)))
  return ticks
}

const geometry = computed(() => {
  const xs = props.x
  const all = props.series.flatMap((s) => s.values)
  if (props.band) all.push(...props.band.lower, ...props.band.upper)
  const finite = all.filter(Number.isFinite)
  if (xs.length === 0 || finite.length === 0) return null

  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  let yMin = Math.min(...finite)
  let yMax = Math.max(...finite)
  const yPad = (yMax - yMin || Math.abs(yMax) || 1) * 0.06
  yMin -= yPad
  yMax += yPad

  const innerW = width.value - pad.left - pad.right
  const innerH = props.height - pad.top - pad.bottom
  const sx = (v: number) => pad.left + ((v - xMin) / (xMax - xMin || 1)) * innerW
  const sy = (v: number) => pad.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH

  const lines = props.series.map((s) => ({
    ...s,
    d: s.values.map((v, i) => `${i === 0 ? "M" : "L"}${sx(xs[i]!).toFixed(1)},${sy(v).toFixed(1)}`).join(""),
  }))

  let bandPath = ""
  if (props.band) {
    const upper = props.band.upper.map((v, i) => `${i === 0 ? "M" : "L"}${sx(xs[i]!).toFixed(1)},${sy(v).toFixed(1)}`)
    const lower = props.band.lower
      .map((v, i) => `L${sx(xs[i]!).toFixed(1)},${sy(v).toFixed(1)}`)
      .reverse()
    bandPath = `${upper.join("")}${lower.join("")}Z`
  }

  const yTicks = niceTicks(yMin, yMax, 5).map((v) => ({ v, y: sy(v) }))
  const xTicks = niceTicks(xMin, xMax, Math.max(2, Math.floor(innerW / 90))).map((v) => ({ v, x: sx(v) }))

  return { sx, sy, lines, bandPath, yTicks, xTicks, innerW, innerH, xMin, xMax }
})

const hoverIndex = ref<number | null>(null)

function indexAt(event: MouseEvent): number | null {
  const g = geometry.value
  if (!g || !container.value) return null
  const rect = container.value.getBoundingClientRect()
  const px = event.clientX - rect.left
  const t = (px - pad.left) / g.innerW
  const target = g.xMin + t * (g.xMax - g.xMin)
  // x is sorted ascending (generations) -- nearest by binary search.
  let lo = 0
  let hi = props.x.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (props.x[mid]! < target) lo = mid + 1
    else hi = mid
  }
  if (lo > 0 && Math.abs(props.x[lo - 1]! - target) < Math.abs(props.x[lo]! - target)) lo -= 1
  return lo
}

function onMove(event: MouseEvent) {
  hoverIndex.value = indexAt(event)
}

function onClick(event: MouseEvent) {
  if (!props.clickable) return
  const i = indexAt(event)
  if (i !== null) emit("select", props.x[i]!)
}

const tooltip = computed(() => {
  const g = geometry.value
  const i = hoverIndex.value
  if (!g || i === null) return null
  const px = g.sx(props.x[i]!)
  return {
    x: px,
    flip: px > width.value * 0.65,
    label: props.x[i],
    rows: props.series.map((s) => ({ label: s.label, color: s.color, value: s.values[i]!, y: g.sy(s.values[i]!) })),
  }
})

const markerX = computed(() => (props.marker !== null && geometry.value ? geometry.value.sx(props.marker) : null))
</script>

<template>
  <div ref="container" class="relative w-full select-none" :style="{ height: `${height}px` }">
    <svg
      v-if="geometry"
      :width="width"
      :height="height"
      class="block"
      :class="clickable ? 'cursor-crosshair' : ''"
      @mousemove="onMove"
      @mouseleave="hoverIndex = null"
      @click="onClick"
    >
      <g class="text-[10px]">
        <g v-for="t in geometry.yTicks" :key="`y${t.v}`">
          <line :x1="pad.left" :x2="width - pad.right" :y1="t.y" :y2="t.y" stroke="#222837" stroke-dasharray="2 4" />
          <text :x="pad.left - 8" :y="t.y" text-anchor="end" dominant-baseline="middle" class="num fill-fg-subtle">
            {{ format(t.v) }}
          </text>
        </g>
        <g v-for="t in geometry.xTicks" :key="`x${t.v}`">
          <line :x1="t.x" :x2="t.x" :y1="height - pad.bottom" :y2="height - pad.bottom + 4" stroke="#323a4d" />
          <text :x="t.x" :y="height - pad.bottom + 16" text-anchor="middle" class="num fill-fg-subtle">{{ t.v }}</text>
        </g>
        <line :x1="pad.left" :x2="width - pad.right" :y1="height - pad.bottom" :y2="height - pad.bottom" stroke="#323a4d" />
        <text :x="width - pad.right" :y="height - 2" text-anchor="end" class="fill-fg-subtle">{{ xLabel }}</text>
      </g>

      <path v-if="band" :d="geometry.bandPath" :fill="band.color" />

      <path
        v-for="line in geometry.lines"
        :key="line.key"
        :d="line.d"
        fill="none"
        :stroke="line.color"
        :stroke-width="line.width ?? 2"
        :stroke-dasharray="line.dashed ? '4 4' : undefined"
        stroke-linejoin="round"
        stroke-linecap="round"
      />

      <g v-if="markerX !== null">
        <line :x1="markerX" :x2="markerX" :y1="pad.top" :y2="height - pad.bottom" stroke="#fbbf24" stroke-width="1.5" />
        <circle :cx="markerX" :cy="pad.top" r="3" fill="#fbbf24" />
      </g>

      <g v-if="tooltip">
        <line :x1="tooltip.x" :x2="tooltip.x" :y1="pad.top" :y2="height - pad.bottom" stroke="#a0a8ba" stroke-opacity="0.4" />
        <circle v-for="row in tooltip.rows" :key="row.label" :cx="tooltip.x" :cy="row.y" r="3.5" :fill="row.color" stroke="#090b10" stroke-width="1.5" />
      </g>
    </svg>

    <div v-else class="flex h-full items-center justify-center text-sm text-fg-subtle">
      waiting for the first generation...
    </div>

    <div
      v-if="tooltip"
      class="pointer-events-none absolute top-2 z-10 min-w-40 rounded-lg border border-line-strong bg-raised/95 px-3 py-2 text-xs shadow-xl backdrop-blur"
      :style="tooltip.flip ? { right: `${width - tooltip.x + 12}px` } : { left: `${tooltip.x + 12}px` }"
    >
      <p class="mb-1 font-mono text-fg-subtle">{{ xLabel }} {{ tooltip.label }}</p>
      <p v-for="row in tooltip.rows" :key="row.label" class="flex items-center justify-between gap-4">
        <span class="flex items-center gap-1.5 text-fg-muted">
          <span class="size-2 rounded-full" :style="{ background: row.color }" />{{ row.label }}
        </span>
        <span class="num text-fg">{{ format(row.value) }}</span>
      </p>
      <p v-if="clickable" class="mt-1 text-[10px] text-fg-subtle">click to watch this generation</p>
    </div>
  </div>
</template>
