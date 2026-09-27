<script setup lang="ts">
import type { RunGroup } from "~/composables/useRunsTable"

// An experiment in the runs table: one collapsible row standing for all its runs (arms x seeds).
defineProps<{ group: RunGroup; open: boolean; now: number }>()
const emit = defineEmits<{ toggle: [] }>()
</script>

<template>
  <tr
    class="cursor-pointer border-b border-line/60 bg-raised/30 transition hover:bg-raised/70"
    :aria-expanded="open"
    data-experiment-group
    @click="emit('toggle')"
  >
    <td class="px-4 py-3">
      <div class="flex items-center gap-2.5">
        <span class="inline-block w-3 text-[10px] text-fg-subtle transition" :class="open ? 'rotate-90' : ''">▶</span>
        <div>
          <p class="font-medium text-fg">{{ group.experiment }}</p>
          <p class="mt-0.5 font-mono text-[11px] text-fg-subtle">
            experiment · {{ group.rows.length }}<template v-if="group.rows.length !== group.total"> of {{ group.total }}</template> runs
          </p>
        </div>
      </div>
    </td>
    <td class="px-3 py-3">
      <StatusBadge :status="group.status" />
      <p class="mt-1 text-[10px] whitespace-nowrap text-fg-subtle">{{ group.statusCounts }}</p>
    </td>
    <td class="px-3 py-3">
      <div class="flex max-w-56 flex-wrap gap-1">
        <span v-for="arm in group.arms.slice(0, 4)" :key="arm" class="chip">{{ arm }}</span>
        <span v-if="group.arms.length > 4" class="text-[11px] text-fg-subtle">+{{ group.arms.length - 4 }} arms</span>
      </div>
    </td>
    <td class="num px-3 py-3 text-right text-fg-subtle">--</td>
    <td class="px-3 py-3 text-xs text-fg-muted">{{ group.algorithms.join(" · ") }}</td>
    <td class="num px-3 py-3 text-xs text-fg-muted">{{ group.generations.toLocaleString() }} in all</td>
    <td class="num px-3 py-3 text-right text-fg-muted">{{ formatFitness(group.best, 3) }}</td>
    <td class="px-3 py-3 text-right">
      <span class="cursor-help text-fg-subtle" title="Comparison runs are aggregated in their experiment, not ranked individually">--</span>
    </td>
    <td class="px-3 py-3" />
    <td class="px-3 py-3 whitespace-nowrap text-fg-muted" :title="`latest run ${formatTimestamp(group.latest)}`">
      {{ formatRelative(group.latest, now) }}
    </td>
    <td class="px-3 py-3" />
    <td class="px-4 py-3 text-right">
      <button class="btn-ghost btn-sm" @click.stop="emit('toggle')">{{ open ? "Hide" : "Show" }} {{ group.rows.length }}</button>
    </td>
  </tr>
</template>
