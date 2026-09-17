<script setup lang="ts">
import type { GenerationStats } from "~/types/telemetry"

const props = defineProps<{ history: GenerationStats[] }>()

const width = 640
const height = 240
const padding = 32

// A hand-rolled SVG polyline rather than a charting library dependency -- this skeleton only needs
// two lines against generation, and it keeps apps/frontend's dependency footprint to exactly what
// doc 0005 named (Nuxt/Vue/Pinia/Tailwind), not an undiscussed addition.
const points = computed(() => {
  const data = props.history
  if (data.length === 0) return { best: "", mean: "" }

  const xs = data.map((d) => d.generation)
  const ys = data.flatMap((d) => [d.best_fitness, d.mean_fitness])
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const xSpan = xMax - xMin || 1
  const ySpan = yMax - yMin || 1

  const scaleX = (x: number) => padding + ((x - xMin) / xSpan) * (width - 2 * padding)
  const scaleY = (y: number) => height - padding - ((y - yMin) / ySpan) * (height - 2 * padding)

  const toPolyline = (key: "best_fitness" | "mean_fitness") =>
    data.map((d) => `${scaleX(d.generation)},${scaleY(d[key])}`).join(" ")

  return { best: toPolyline("best_fitness"), mean: toPolyline("mean_fitness") }
})
</script>

<template>
  <svg :viewBox="`0 0 ${width} ${height}`" class="w-full rounded-lg border border-slate-200 bg-white">
    <template v-if="history.length > 0">
      <polyline :points="points.mean" fill="none" stroke="#94a3b8" stroke-width="1.5" />
      <polyline :points="points.best" fill="none" stroke="#2563eb" stroke-width="2" />
      <text x="8" y="16" class="fill-blue-600 text-xs">best fitness</text>
      <text x="8" y="30" class="fill-slate-400 text-xs">mean fitness</text>
    </template>
    <text v-else x="50%" y="50%" text-anchor="middle" class="fill-slate-400 text-sm">
      waiting for the first generation...
    </text>
  </svg>
</template>
