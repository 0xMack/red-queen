<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"

// Live "race" while a human plays: every leaderboard entrant sits on a track at its mean held-out
// score, and the player's marker slides along as they eat -- lighting up each algorithm they pass.
const props = defineProps<{ entries: EvaluationRecord[]; score: number }>()

const entrants = computed(() =>
  [...props.entries]
    .sort((a, b) => a.metrics.quality.mean - b.metrics.quality.mean)
    .map((r) => ({ id: r.entrant_id, label: entrantShortLabel(r), mean: r.metrics.quality.mean, color: entrantColor(r) })),
)
const max = computed(() => Math.max(5, props.score + 2, ...entrants.value.map((m) => m.mean)) * 1.08)

// Entrants within ~4% of the track of each other share one marker ("5 entrants") -- otherwise
// e.g. every near-zero model stacks its label on the same spot.
const markers = computed(() => {
  const groups: { id: string; label: string; mean: number; color: string; members: string[] }[] = []
  for (const e of entrants.value) {
    const last = groups.at(-1)
    if (last && (e.mean - last.mean) / max.value < 0.04) {
      last.members.push(`${e.label} (${e.mean.toFixed(1)})`)
      last.mean = Math.max(last.mean, e.mean)
      last.label = `${last.members.length} entrants`
    } else {
      groups.push({ id: e.id, label: e.label, mean: e.mean, color: e.color, members: [`${e.label} (${e.mean.toFixed(1)})`] })
    }
  }
  return groups
})
const pct = (v: number) => `${(v / max.value) * 100}%`

const passed = computed(() => entrants.value.filter((e) => props.score > e.mean))
const next = computed(() => markers.value.find((m) => m.mean >= props.score) ?? null)
</script>

<template>
  <div class="rounded-xl border border-line bg-sunken p-4">
    <div class="flex items-baseline justify-between">
      <p class="text-[11px] tracking-wide text-fg-subtle uppercase">Live vs. the algorithms</p>
      <p class="text-xs text-fg-muted">
        passed <span class="num font-semibold text-queen-200">{{ passed.length }}</span> / {{ entrants.length }}
      </p>
    </div>

    <!-- The track: entrants at their mean score, the player sliding along it. -->
    <div class="relative mt-10 mb-10 h-2 rounded-full bg-raised">
      <div class="absolute inset-y-0 left-0 rounded-full bg-queen-500/40 transition-[width] duration-300" :style="{ width: pct(score) }" />
      <div
        v-for="(m, i) in markers"
        :key="m.id"
        class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
        :style="{ left: pct(m.mean) }"
      >
        <span
          class="block size-3 rounded-full border-2 transition-all duration-300"
          :style="score > m.mean ? { background: m.color, borderColor: m.color, boxShadow: `0 0 10px ${m.color}` } : { borderColor: m.color, background: '#0b0e14' }"
        />
        <span
          class="absolute left-1/2 w-max max-w-28 -translate-x-1/2 truncate text-[10px] transition-colors"
          :class="[i % 2 ? 'top-5' : 'bottom-5', score > m.mean ? 'text-fg' : 'text-fg-subtle']"
          :title="m.members.join(', ')"
        >
          {{ m.label }}
        </span>
      </div>
      <div class="absolute top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 transition-[left] duration-300 ease-out" :style="{ left: pct(score) }">
        <span class="num flex h-6 min-w-6 items-center justify-center rounded-full bg-queen-400 px-1.5 text-xs font-bold text-white shadow-[0_0_16px_rgb(255_92_122/0.9)]">
          {{ score }}
        </span>
      </div>
    </div>

    <p class="text-sm text-fg-muted">
      <template v-if="next">
        Next up: <span class="text-fg">{{ next.label }}</span> {{ next.members.length > 1 ? "average up to" : "averages" }}
        <span class="num text-fg">{{ next.mean.toFixed(1) }}</span>
        <span class="text-fg-subtle"> ({{ Math.max(1, Math.floor(next.mean) + 1 - score) }} more)</span>
      </template>
      <template v-else><span class="font-semibold text-queen-200">Ahead of every algorithm's average.</span></template>
    </p>
  </div>
</template>
