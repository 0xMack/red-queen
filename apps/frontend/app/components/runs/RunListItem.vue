<script setup lang="ts">
import type { RunInfo } from "~/types/telemetry"

// One run as a compact, linked row: name, id and age, its training trend, best value, status.
const props = defineProps<{ run: RunInfo; trend?: number[]; best?: number | null; now: number }>()
const meta = computed(() => describeRun(props.run))
</script>

<template>
  <NuxtLink :to="`/runs/${run.run_id}`" class="group flex items-center gap-4 px-4 py-3 transition hover:bg-raised/60">
    <div class="min-w-0 flex-1">
      <p class="truncate text-sm font-medium transition group-hover:text-queen-200">{{ meta.title }}</p>
      <p class="font-mono text-[11px] text-fg-subtle">{{ shortId(run.run_id) }} · {{ formatRelative(run.created_at, now) }}</p>
    </div>
    <Sparkline :values="trend ?? []" class="hidden h-7 w-24 sm:block" />
    <span class="num w-16 text-right text-sm">{{ formatFitness(best ?? null, 2) }}</span>
    <StatusBadge :status="run.status" :stale="isStale(run, now)" />
  </NuxtLink>
</template>
