<script setup lang="ts">
import type { GameEntry } from "~/data/games"
import type { ScoreSpec } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"

// One game's headline result, live from its leaderboard (docs/design/0007): who leads, by how much, and against what
// -- the best hand-written baseline and random play, drawn to scale -- so a number means something. The landing
// page's hero shows one per game. Links to the game.
const props = defineProps<{ game: GameEntry; entries: EvaluationRecord[]; score: ScoreSpec }>()

const leader = computed(() => props.entries[0] ?? null)
const baselines = computed(() => props.entries.filter((r) => r.entrant_kind === "baseline"))
// The strongest baseline that isn't the leader itself; random is the floor.
const bestBaseline = computed(() => baselines.value.find((r) => r !== leader.value && baselineName(r) !== "random") ?? null)
const random = computed(() => baselines.value.find((r) => baselineName(r) === "random") ?? null)

const rows = computed(() =>
  [
    leader.value && { record: leader.value, role: "leader" as const },
    bestBaseline.value && { record: bestBaseline.value, role: "baseline" as const },
    random.value && { record: random.value, role: "random" as const },
  ].filter((r): r is { record: EvaluationRecord; role: "leader" | "baseline" | "random" } => !!r),
)
const max = computed(() => Math.max(props.score.scaleMin, ...rows.value.map((r) => r.record.metrics.quality.mean)))
const width = (v: number) => `${Math.max(1.5, (Math.max(0, v) / max.value) * 100)}%`

// The claim, in words: how far ahead of the best baseline (a ratio where scores are counts, a margin otherwise).
const claim = computed(() => {
  const l = leader.value
  const b = bestBaseline.value
  if (!l || !b) return null
  if (l.entrant_kind === "baseline") return "A hand-written baseline still leads."
  const lead = l.metrics.quality.mean
  const base = b.metrics.quality.mean
  if (props.score.scaleMin > 1 || base <= 0) return `${props.score.format(lead - base)} ahead of the best baseline.`
  const ratio = lead / base
  return ratio >= 1.5 ? `${ratio.toFixed(1)}× the best baseline.` : "Ahead of the best baseline."
})
</script>

<template>
  <NuxtLink :to="game.href" class="card card-hover group flex min-w-0 flex-col p-4 sm:p-5">
    <div class="flex items-baseline justify-between gap-3">
      <p class="label"><span class="text-queen-300">{{ game.title }}</span> · {{ score.label }}</p>
      <span class="text-xs text-fg-subtle transition group-hover:text-fg" aria-hidden="true">→</span>
    </div>
    <template v-if="leader">
      <div class="mt-3 flex items-baseline justify-between gap-3">
        <p class="min-w-0 truncate text-[15px] font-semibold text-fg" :title="leader.label">{{ entrantShortLabel(leader) }}</p>
        <p class="num shrink-0 text-2xl text-fg">{{ score.format(leader.metrics.quality.mean) }}</p>
      </div>
      <ul class="mt-3 space-y-1.5">
        <li v-for="r in rows" :key="r.record.entrant_id" class="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-3">
          <div>
            <div class="h-1.5 rounded-full bg-sunken">
              <div
                class="h-full rounded-full"
                :style="{
                  width: width(r.record.metrics.quality.mean),
                  background: r.role === 'leader' ? entrantColor(r.record) : r.role === 'baseline' ? palette.gold400 : palette.lineStrong,
                }"
              />
            </div>
            <p v-if="r.role !== 'leader'" class="mt-0.5 truncate font-mono text-[10px] text-fg-subtle">{{ r.record.label.toLowerCase() }}</p>
          </div>
          <span class="num text-right text-xs text-fg-subtle">{{ r.role === "leader" ? "" : score.format(r.record.metrics.quality.mean) }}</span>
        </li>
      </ul>
      <p v-if="claim" class="mt-auto pt-3 text-xs text-fg-muted">{{ claim }}</p>
    </template>
    <p v-else class="mt-3 text-sm text-fg-subtle">No leaderboard yet -- play it yourself.</p>
  </NuxtLink>
</template>
