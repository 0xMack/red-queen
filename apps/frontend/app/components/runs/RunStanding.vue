<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"

// A run's place on its game's leaderboard (docs/design/0007): the held-out game score -- what training fitness is
// *not* -- as a banner linking to the entrant on the game page.
defineProps<{ record: EvaluationRecord; rank: number; of: number }>()
</script>

<template>
  <NuxtLink
    :to="{ path: `/games/${record.game}`, query: { watch: record.entrant_id } }"
    class="card card-hover group flex flex-wrap items-center gap-x-7 gap-y-2 px-5 py-3.5 text-sm"
  >
    <span class="label text-queen-300">Leaderboard</span>
    <span class="font-display text-2xl leading-none">#{{ rank }}<span class="text-base text-fg-subtle"> of {{ of }}</span></span>
    <span class="text-fg-muted">
      held-out score
      <span class="num text-fg">{{ record.metrics.quality.mean.toFixed(2) }}</span>
      <span class="num text-fg-subtle"> ± {{ record.metrics.quality.ci95.toFixed(2) }}</span>
      over {{ record.metrics.quality.n }} unseen games
    </span>
    <span v-if="record.metrics.quality.train_mean !== null" class="text-fg-muted">
      vs. <span class="num text-fg">{{ record.metrics.quality.train_mean.toFixed(1) }}</span> on its training seeds
    </span>
    <span class="text-fg-muted"><span class="num text-fg">{{ record.metrics.inference.total_us.toFixed(1) }}</span> µs / decision</span>
    <span class="ml-auto text-xs text-fg-subtle transition group-hover:text-fg">Training fitness above is not a game score · see it ranked →</span>
  </NuxtLink>
</template>
