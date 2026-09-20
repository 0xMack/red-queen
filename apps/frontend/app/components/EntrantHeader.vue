<script setup lang="ts">
import type { ScoreSpec } from "~/games/types"
import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"

// Who is on the stage, for any game: the entrant's name and colour, where it ranks, the representation
// it sees through, its score with a 95% interval, and a link to the run that trained it. Sits above
// whatever the game's Watch stage shows.
defineProps<{
  entry: EvaluationRecord
  rank: number
  total: number
  score: ScoreSpec
  interfacesById: Record<string, InterfaceInfo>
}>()
</script>

<template>
  <div class="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2">
    <span class="flex items-center gap-2">
      <span class="size-2.5 rounded-full" :style="{ background: entrantColor(entry) }" />
      <span class="font-display text-lg font-semibold" :title="entry.label">{{ entrantShortLabel(entry) }}</span>
    </span>
    <span class="chip">#{{ rank }} of {{ total }}</span>
    <span class="chip" :style="{ color: LEVEL_COLORS[entry.metrics.model.observer_level] }">
      L{{ entry.metrics.model.observer_level }} {{ interfacesById[entry.interface]?.observer.level_name }}
    </span>
    <span class="text-sm text-fg-muted">
      {{ score.label.toLowerCase() }}
      <span class="num font-semibold text-fg">{{ score.format(entry.metrics.quality.mean) }}</span>
      <span class="num text-fg-subtle"> ± {{ score.format(entry.metrics.quality.ci95) }}</span>
    </span>
    <NuxtLink v-if="entry.run_id" :to="`/runs/${entry.run_id}`" class="link ml-auto text-sm">Training run →</NuxtLink>
  </div>
</template>
