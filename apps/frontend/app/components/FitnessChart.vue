<script setup lang="ts">
import type { GenerationStats } from "~/types/telemetry"

// Best/mean fitness vs. generation, with the population's full worst..best spread shaded behind --
// the spread collapsing is the visual tell for premature convergence. Built on LineChart.vue; used
// by the live run page and (statically) by RunFitnessPreview in the Learn chapters.
const props = withDefaults(
  defineProps<{ history: GenerationStats[]; height?: number; marker?: number | null; clickable?: boolean; xLabel?: string }>(),
  { height: 300, marker: null, clickable: false },
)
defineEmits<{ select: [generation: number] }>()

const x = computed(() => props.history.map((h) => h.generation))
const series = computed(() => [
  { key: "mean", label: "mean", color: "#60a5fa", values: props.history.map((h) => h.mean_fitness), width: 1.5 },
  { key: "best", label: "best", color: "#ff5c7a", values: props.history.map((h) => h.best_fitness), width: 2.25 },
])
const band = computed(() => ({
  lower: props.history.map((h) => h.worst_fitness),
  upper: props.history.map((h) => h.best_fitness),
  color: "rgb(255 92 122 / 0.08)",
}))
</script>

<template>
  <div>
    <div class="mb-2 flex flex-wrap items-center gap-4 text-xs text-fg-muted">
      <span class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded bg-queen-400" />best</span>
      <span class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded bg-signal-400" />mean</span>
      <span class="flex items-center gap-1.5"><span class="h-2.5 w-4 rounded-sm bg-queen-400/15" />worst..best spread</span>
    </div>
    <LineChart :x-label="xLabel ?? 'generation'"
      :x="x"
      :series="series"
      :band="band"
      :height="height"
      :marker="marker"
      :clickable="clickable"
      @select="$emit('select', $event)"
    />
  </div>
</template>
