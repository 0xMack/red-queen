<script setup lang="ts">
import type { ParetoPoint } from "~/types/chart"
import type { EvaluationRecord } from "~/types/leaderboard"

// Score vs. a selectable cost axis for every entrant (docs/design/0007: tradeoffs, not one blended
// score). Clicking a point selects that entrant.
const props = defineProps<{ entries: EvaluationRecord[] }>()
defineEmits<{ select: [entrantId: string] }>()

type CostKey = "inference" | "parameters" | "episodes" | "trainTime"
// Baselines count as zero training cost and zero parameters.
const COST_AXES: Record<CostKey, { label: string; value: (r: EvaluationRecord) => number | null; format: (v: number) => string }> = {
  inference: { label: "inference µs / decision", value: (r) => r.metrics.inference.total_us, format: (v) => `${v}` },
  parameters: { label: "parameters", value: (r) => r.metrics.inference.parameters, format: compactNumber },
  episodes: {
    label: "training episodes",
    value: (r) => (r.metrics.training.none ? 0 : (r.metrics.training.episodes ?? null)),
    format: compactNumber,
  },
  trainTime: {
    label: "training time (s)",
    value: (r) => (r.metrics.training.none ? 0 : (r.metrics.training.active_s ?? null)),
    format: compactNumber,
  },
}
const costKey = ref<CostKey>("inference")

const points = computed<ParetoPoint[]>(() =>
  props.entries.flatMap((r) => {
    const x = COST_AXES[costKey.value].value(r)
    return x === null
      ? []
      : [{ id: r.entrant_id, label: entrantLabel(r), x, y: r.metrics.quality.mean, err: r.metrics.quality.ci95, color: entrantColor(r) }]
  }),
)
const unplotted = computed(() => props.entries.length - points.value.length)
</script>

<template>
  <section class="card p-5">
    <div class="flex flex-wrap items-baseline justify-between gap-3">
      <div>
        <h3 class="text-lg font-semibold">Score vs. cost</h3>
        <p class="mt-1 text-xs text-fg-subtle">
          The dashed line is the Pareto front: entrants nothing else beats on both score and this cost.
          Click a point to watch it.
        </p>
      </div>
      <div class="flex flex-wrap rounded-lg border border-line bg-sunken p-0.5">
        <button
          v-for="(axis, key) in COST_AXES"
          :key="key"
          class="rounded-md px-3 py-1.5 text-xs transition"
          :class="costKey === key ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
          @click="costKey = key"
        >
          {{ axis.label }}
        </button>
      </div>
    </div>
    <ClientOnly>
      <ParetoChart
        class="mt-4"
        :points="points"
        :x-label="COST_AXES[costKey].label"
        :format-x="COST_AXES[costKey].format"
        :height="320"
        @select="$emit('select', $event)"
      />
    </ClientOnly>
    <div class="mt-3 flex flex-wrap items-center gap-4 text-[11px] text-fg-subtle">
      <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-full bg-gold-400" />baseline</span>
      <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-full" :style="{ background: LEVEL_COLORS[3] }" />L3 engineered</span>
      <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-full" :style="{ background: LEVEL_COLORS[1] }" />L1 full state</span>
      <span v-if="unplotted" class="ml-auto">{{ unplotted }} entrant(s) without this cost measured aren't plotted.</span>
    </div>
  </section>
</template>
