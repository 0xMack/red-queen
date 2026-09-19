<script setup lang="ts">
import type { ParetoPoint } from "~/types/chart"

// Score (higher is better) against one cost axis (lower is better), with 95% intervals as error
// bars and the Pareto front -- entries nothing else beats on *both* axes -- highlighted
// (docs/design/0007: "no single weighted score"). Costs span orders of magnitude (0 parameters for
// a baseline, thousands for a grid network), so x is symlog: log10(1 + x), which keeps 0 plottable.

const props = withDefaults(
  defineProps<{ points: ParetoPoint[]; xLabel: string; yLabel?: string; formatX?: (v: number) => string; height?: number }>(),
  { yLabel: "mean score", formatX: (v: number) => String(v), height: 360 },
)
const emit = defineEmits<{ select: [id: string] }>()

const container = ref<HTMLElement | null>(null)
const width = ref(720)
let observer: ResizeObserver | null = null
onMounted(() => {
  observer = new ResizeObserver((entries) => {
    const w = entries[0]?.contentRect.width
    if (w) width.value = Math.max(280, Math.round(w))
  })
  if (container.value) observer.observe(container.value)
})
onUnmounted(() => observer?.disconnect())

const pad = { top: 20, right: 24, bottom: 42, left: 52 }
const tx = (v: number) => Math.log10(1 + Math.max(0, v))

const frontIds = computed(() => {
  const ids = new Set<string>()
  for (const p of props.points) {
    const dominated = props.points.some(
      (q) => q !== p && q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y),
    )
    if (!dominated) ids.add(p.id)
  }
  return ids
})

const geometry = computed(() => {
  if (props.points.length === 0) return null
  const xs = props.points.map((p) => tx(p.x))
  const xMin = Math.min(0, ...xs)
  const xMax = Math.max(...xs) * 1.08 || 1
  const yMax = Math.max(...props.points.map((p) => p.y + p.err)) * 1.1 || 1
  const yMin = Math.min(0, ...props.points.map((p) => p.y - p.err))
  const innerW = width.value - pad.left - pad.right
  const innerH = props.height - pad.top - pad.bottom
  const sx = (v: number) => pad.left + ((tx(v) - xMin) / (xMax - xMin || 1)) * innerW
  const sy = (v: number) => pad.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH

  // Ticks at 0 and powers of ten within range.
  const xTicks = [0]
  for (let e = 1; tx(10 ** e) <= xMax; e++) xTicks.push(10 ** e)
  const yStep = yMax > 20 ? 5 : yMax > 8 ? 2 : yMax > 3 ? 1 : 0.5
  const yTicks: number[] = []
  for (let v = Math.ceil(yMin / yStep) * yStep; v <= yMax; v += yStep) yTicks.push(Number(v.toFixed(2)))

  const front = props.points
    .filter((p) => frontIds.value.has(p.id))
    .sort((a, b) => a.x - b.x)
    .map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
    .join(" ")

  return { sx, sy, xTicks, yTicks, front, innerH }
})

// Only the Pareto front is labelled (everything else: hover, or the table) -- long entrant names on
// a crowded chart overprint each other. Front entrants that land on (almost) the same spot, e.g.
// two identical champions, share one label with a "+N" count.
const labels = computed(() => {
  const g = geometry.value
  const out: Record<string, string> = {}
  if (!g) return out
  const placed: { x: number; y: number; id: string; extra: number }[] = []
  for (const p of props.points.filter((q) => frontIds.value.has(q.id))) {
    const x = g.sx(p.x)
    const y = g.sy(p.y)
    const near = placed.find((q) => Math.abs(q.x - x) < 14 && Math.abs(q.y - y) < 12)
    if (near) near.extra += 1
    else placed.push({ x, y, id: p.id, extra: 0 })
  }
  for (const q of placed) {
    const label = props.points.find((p) => p.id === q.id)!.label
    out[q.id] = q.extra ? `${label}  +${q.extra} more` : label
  }
  return out
})

const hovered = ref<string | null>(null)
const hoveredPoint = computed(() => props.points.find((p) => p.id === hovered.value) ?? null)
</script>

<template>
  <div ref="container" class="relative w-full" :style="{ height: `${height}px` }">
    <svg v-if="geometry" :width="width" :height="height" class="block">
      <g class="text-[10px]">
        <g v-for="t in geometry.yTicks" :key="`y${t}`">
          <line :x1="pad.left" :x2="width - pad.right" :y1="geometry.sy(t)" :y2="geometry.sy(t)" stroke="#222837" stroke-dasharray="2 4" />
          <text :x="pad.left - 8" :y="geometry.sy(t)" text-anchor="end" dominant-baseline="middle" class="num fill-fg-subtle">{{ t }}</text>
        </g>
        <g v-for="t in geometry.xTicks" :key="`x${t}`">
          <line :x1="geometry.sx(t)" :x2="geometry.sx(t)" :y1="pad.top" :y2="height - pad.bottom" stroke="#1b202c" />
          <text :x="geometry.sx(t)" :y="height - pad.bottom + 16" text-anchor="middle" class="num fill-fg-subtle">{{ formatX(t) }}</text>
        </g>
        <text :x="width - pad.right" :y="height - 6" text-anchor="end" class="fill-fg-muted">{{ xLabel }} → (log scale, lower is cheaper)</text>
        <text :x="pad.left" :y="12" class="fill-fg-muted">↑ {{ yLabel }} (95% CI)</text>
      </g>

      <polyline
        v-if="geometry.front"
        :points="geometry.front"
        fill="none"
        stroke="#ff5c7a"
        stroke-width="1.5"
        stroke-dasharray="5 4"
        opacity="0.8"
      />

      <g
        v-for="p in points"
        :key="p.id"
        class="cursor-pointer"
        @mouseenter="hovered = p.id"
        @mouseleave="hovered = null"
        @click="emit('select', p.id)"
      >
        <line
          :x1="geometry.sx(p.x)"
          :x2="geometry.sx(p.x)"
          :y1="geometry.sy(p.y - p.err)"
          :y2="geometry.sy(p.y + p.err)"
          :stroke="p.color"
          stroke-opacity="0.6"
          stroke-width="2"
        />
        <circle
          :cx="geometry.sx(p.x)"
          :cy="geometry.sy(p.y)"
          :r="frontIds.has(p.id) ? 7 : 5"
          :fill="p.color"
          :stroke="frontIds.has(p.id) ? '#ff8fa3' : '#090b10'"
          :stroke-width="frontIds.has(p.id) ? 2.5 : 1.5"
        />
        <text
          v-if="labels[p.id]"
          :x="geometry.sx(p.x) + 10"
          :y="geometry.sy(p.y) - 8"
          class="text-[11px]"
          :class="frontIds.has(p.id) ? 'fill-fg' : 'fill-fg-subtle'"
        >
          {{ labels[p.id] }}
        </text>
      </g>
    </svg>
    <div v-else class="flex h-full items-center justify-center text-sm text-fg-subtle">No entrants with this cost measured.</div>

    <div
      v-if="hoveredPoint && geometry"
      class="pointer-events-none absolute z-10 rounded-lg border border-line-strong bg-raised/95 px-3 py-2 text-xs shadow-xl"
      :style="{ left: `${Math.min(geometry.sx(hoveredPoint.x) + 14, width - 220)}px`, top: `${geometry.sy(hoveredPoint.y) + 10}px` }"
    >
      <p class="font-medium text-fg">{{ hoveredPoint.label }}</p>
      <p class="num text-fg-muted">{{ yLabel }} {{ hoveredPoint.y.toFixed(2) }} ± {{ hoveredPoint.err.toFixed(2) }}</p>
      <p class="num text-fg-muted">{{ xLabel }} {{ formatX(hoveredPoint.x) }}</p>
      <p v-if="frontIds.has(hoveredPoint.id)" class="mt-1 text-queen-300">On the Pareto front</p>
    </div>
  </div>
</template>
