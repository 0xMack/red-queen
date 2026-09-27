<script setup lang="ts">
import type { RunRow } from "~/utils/runMeta"

// One run in the runs page's card list (below lg): its own card, or one of an experiment group's.
defineProps<{ r: RunRow; now: number }>()
</script>

<template>
  <NuxtLink :to="`/runs/${r.run.run_id}`" class="card card-hover block p-4">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <p class="truncate font-medium">{{ r.meta.title }}</p>
        <p class="font-mono text-[11px] text-fg-subtle">{{ shortId(r.run.run_id) }} · {{ formatRelative(r.run.created_at, now) }}</p>
      </div>
      <StatusBadge :status="r.run.status" :stale="r.stale" />
    </div>
    <Sparkline :values="r.trend" class="mt-3 h-10 w-full" />
    <dl class="mt-3 grid grid-cols-3 gap-2 text-xs">
      <div><dt class="text-fg-subtle">best</dt><dd class="num text-fg">{{ formatFitness(r.best, 2) }}</dd></div>
      <div>
        <dt class="text-fg-subtle">gens</dt>
        <dd class="num text-fg">{{ r.generations }}<span v-if="r.meta.targetGenerations" class="text-fg-subtle">/{{ r.meta.targetGenerations }}</span></dd>
      </div>
      <div><dt class="text-fg-subtle">duration</dt><dd class="num text-fg">{{ formatDuration(r.duration) }}</dd></div>
    </dl>
    <p v-if="r.meta.selection" class="mt-3 flex flex-wrap gap-1.5">
      <span class="chip">{{ r.meta.selection }}</span>
      <span v-if="r.meta.populationSize" class="chip">pop {{ r.meta.populationSize }}</span>
    </p>
  </NuxtLink>
</template>
