<script setup lang="ts">
import type { RunGroup } from "~/composables/useRunsTable"

// An experiment as a card, for narrow screens (the table's `RunsExperimentRow`).
defineProps<{ group: RunGroup; open: boolean; now: number }>()
const emit = defineEmits<{ toggle: [] }>()
</script>

<template>
  <button class="card card-hover block p-4 text-left" :aria-expanded="open" data-experiment-group @click="emit('toggle')">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <p class="truncate font-medium">
          <span class="inline-block w-3 text-[10px] text-fg-subtle transition" :class="open ? 'rotate-90' : ''">▶</span>
          {{ group.experiment }}
        </p>
        <p class="text-[11px] text-fg-subtle">experiment · {{ group.total }} runs · {{ group.algorithms.join(" · ") }}</p>
      </div>
      <StatusBadge :status="group.status" />
    </div>
    <UiStats
      class="mt-3"
      :items="[
        { label: 'best', value: formatFitness(group.best, 2) },
        { label: 'gens', value: group.generations.toLocaleString() },
        { label: 'latest', value: formatRelative(group.latest, now) },
      ]"
    />
  </button>
</template>
