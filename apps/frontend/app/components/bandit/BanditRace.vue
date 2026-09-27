<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import type { BanditRunState } from "~/composables/useBanditRun"

// Players on the same game, side by side (docs/design/0011): every lane is one player's pulls, split by machine --
// the colours show at a glance who found the good machine and who kept exploring -- and what they have won. Skill
// (how much better than random, from the true values) stays hidden until the game ends: during a person's game it
// would give the best machine away. Afterwards, skill over the game for everyone.
export interface Racer {
  id: string
  label: string
  run: BanditRunState
  you?: boolean
}

const props = defineProps<{ racers: Racer[]; binary: boolean; revealed: boolean }>()
const SKIP = 9

function split(run: BanditRunState) {
  const counts = run.counts.value
  const total = Math.max(1, run.budget.value)
  return counts.map((c, arm) => ({ arm, c, width: (c / total) * 100 })).filter((s) => s.c > 0)
}

const chart = computed<{ x: number[]; series: ChartSeries[] } | null>(() => {
  if (!props.revealed) return null
  const series = props.racers.map((r, i) => ({
    key: r.id,
    label: r.label,
    color: r.you ? palette.fg : seriesColors[(i + 1) % seriesColors.length]!,
    values: r.run.skillCurve.value.map((s) => Math.round(s * 1000) / 10),
    width: r.you ? 2.5 : 1.5,
  }))
  // From pull 10: over the first few pulls one choice swings skill between -100 and 100, flattening the rest.
  const n = Math.max(...series.map((s) => s.values.length))
  if (n < SKIP + 2) return null
  return { x: Array.from({ length: n - SKIP }, (_, i) => i + SKIP + 1), series: series.map((s) => ({ ...s, values: s.values.slice(SKIP, n) })) }
})
</script>

<template>
  <div>
    <ol class="space-y-2.5">
      <li v-for="r in racers" :key="r.id" class="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_3.5rem_3rem] items-center gap-3 text-xs">
        <span class="truncate" :class="r.you ? 'font-semibold text-fg' : 'text-fg-muted'" :title="r.label">{{ r.label }}</span>
        <span class="flex h-3 overflow-hidden rounded-[3px] bg-sunken">
          <span
            v-for="s in split(r.run)"
            :key="s.arm"
            class="h-full border-r border-bg/60 transition-[width] duration-200 last:border-r-0"
            :style="{ width: `${s.width}%`, background: armColor(s.arm) }"
            :title="`${armName(s.arm)}: ${s.c} pulls`"
          />
        </span>
        <span class="num text-right text-fg">{{ binary ? r.run.total.value : r.run.total.value.toFixed(1) }}</span>
        <span class="num text-right" :class="revealed ? 'text-fg' : 'text-fg-subtle'">{{ revealed ? Math.round(r.run.skill.value * 100) : "··" }}</span>
      </li>
    </ol>
    <p class="mt-2 flex justify-end gap-6 font-mono text-[10px] text-fg-subtle"><span>won</span><span>skill</span></p>
    <template v-if="chart">
      <p class="label mt-4 mb-1">Skill over the game</p>
      <UiLegend :series="chart.series" />
      <LineChart class="mt-1" :x="chart.x" :series="chart.series" :height="150" x-label="pull" :format="(v: number) => v.toFixed(0)" />
    </template>
  </div>
</template>
