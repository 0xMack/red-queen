<script setup lang="ts">
import type { ScoreSpec } from "~/games/types"
import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"

// Who is on the stage, for any game: the entrant's name and colour, where it ranks, the representation
// it sees through, its score with a 95% interval, and a link to the run that trained it. The stage panel's caption.
defineProps<{
  entry: EvaluationRecord
  rank: number
  total: number
  score: ScoreSpec
  interfacesById: Record<string, InterfaceInfo>
}>()
</script>

<template>
  <div class="flex w-full flex-wrap items-center gap-x-3 gap-y-1.5 py-1">
    <span class="num text-xs text-fg-subtle">#{{ rank }}<span class="text-fg-subtle/60">/{{ total }}</span></span>
    <span class="flex items-center gap-2">
      <span class="size-2.5 rounded-full" :style="{ background: entrantColor(entry) }" />
      <span class="text-[15px] font-semibold" :title="entry.label">{{ entrantShortLabel(entry) }}</span>
    </span>
    <span class="chip" :style="{ color: LEVEL_COLORS[entry.metrics.model.observer_level] }">
      L{{ entry.metrics.model.observer_level }} {{ interfacesById[entry.interface]?.observer.level_name }}
    </span>
    <span class="ml-auto flex items-baseline gap-1.5 text-sm">
      <span class="label">{{ score.label }}</span>
      <span class="num text-base text-fg">{{ score.format(entry.metrics.quality.mean) }}</span>
      <span class="num text-xs text-fg-subtle">± {{ score.format(entry.metrics.quality.ci95) }}</span>
    </span>
    <NuxtLink v-if="entry.run_id" :to="`/runs/${entry.run_id}`" class="btn-quiet btn-sm">Training run →</NuxtLink>
  </div>
</template>
