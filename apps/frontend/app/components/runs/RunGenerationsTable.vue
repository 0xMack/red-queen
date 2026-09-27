<script setup lang="ts">
import type { GenerationStats } from "~/types/telemetry"
import type { RunTerms } from "~/utils/runMeta"

// A run's latest generations (or iterations) as a table: best with its change, mean, worst, diversity, elapsed.
// `clickable` rows pin that generation's champion.
const props = defineProps<{ history: GenerationStats[]; terms: RunTerms; pinned: number | null; clickable: boolean; limit?: number }>()
const emit = defineEmits<{ pin: [generation: number] }>()
const recent = computed(() => props.history.slice(-(props.limit ?? 25)).reverse())
const byGeneration = computed(() => new Map(props.history.map((h) => [h.generation, h])))
const first = computed(() => props.history[0])
function delta(h: GenerationStats): number | null {
  const prev = byGeneration.value.get(h.generation - 1)
  return prev ? h.best_fitness - prev.best_fitness : null
}
</script>

<template>
  <UiPanel pad="none" class="overflow-hidden" :title="`Recent ${terms.unit}s`">
    <template #actions><span class="font-mono text-[11px] text-fg-subtle">latest {{ recent.length }} of {{ history.length }}</span></template>
    <div class="max-h-[420px] overflow-auto">
      <table class="w-full text-sm">
        <thead class="sticky top-0 bg-surface">
          <tr class="label border-b border-line text-left">
            <th class="px-5 py-2 font-medium">{{ capitalize(terms.unit) }}</th>
            <th class="px-3 py-2 text-right font-medium">Best</th>
            <th class="px-3 py-2 text-right font-medium">Δ best</th>
            <th class="px-3 py-2 text-right font-medium">Mean</th>
            <th class="px-3 py-2 text-right font-medium">Worst</th>
            <th class="px-3 py-2 text-right font-medium">{{ capitalize(terms.diversity) }}</th>
            <th class="px-5 py-2 text-right font-medium">Time</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="h in recent"
            :key="h.generation"
            class="border-b border-line/40 last:border-0"
            :class="[clickable ? 'cursor-pointer hover:bg-raised/60' : '', pinned === h.generation ? 'bg-gold-400/10' : '']"
            @click="clickable && emit('pin', h.generation)"
          >
            <td class="num px-5 py-1.5 text-fg-muted">{{ h.generation }}</td>
            <td class="num px-3 py-1.5 text-right text-fg">{{ formatFitness(h.best_fitness, 3) }}</td>
            <td class="num px-3 py-1.5 text-right text-xs">
              <span v-if="(delta(h) ?? 0) > 1e-9" class="text-life-400">{{ formatSigned(delta(h)!, 3) }}</span>
              <span v-else-if="(delta(h) ?? 0) < -1e-9" class="text-queen-300">{{ formatSigned(delta(h)!, 3) }}</span>
              <span v-else class="text-fg-subtle">·</span>
            </td>
            <td class="num px-3 py-1.5 text-right text-fg-muted">{{ formatFitness(h.mean_fitness, 3) }}</td>
            <td class="num px-3 py-1.5 text-right text-fg-subtle">{{ formatFitness(h.worst_fitness, 3) }}</td>
            <td class="num px-3 py-1.5 text-right text-fg-muted">{{ formatFitness(h.diversity, 3) }}</td>
            <td class="num px-5 py-1.5 text-right text-xs text-fg-subtle">{{ first ? `+${formatDuration(h.timestamp - first.timestamp)}` : "" }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </UiPanel>
</template>
