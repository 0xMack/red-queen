<script setup lang="ts">
import type { ScoreSpec } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { DeviceFit } from "~/types/modelpack"

// Compact, rank-ordered leaderboard for the game page's side panel. Doubles as the "what am I
// watching" selector (click an entrant to play it), and -- when `human` is set -- slots the
// player in by score, animating their row up the ranking as they play (TransitionGroup moves rows
// with a FLIP transition whenever the order changes). `runsHere` (docs/design/0009) marks entrants
// this device can't run -- still listed, with the reason on hover, never hidden unless asked.
const props = defineProps<{
  entries: EvaluationRecord[]
  /** How a score prints, and the floor its bars scale to (a game's ScoreSpec). */
  score: ScoreSpec
  selectedId?: string | null
  human?: { score: number; label: string; live: boolean } | null
  runsHere?: Record<string, DeviceFit>
}>()
defineEmits<{ select: [entrantId: string] }>()

interface Row {
  key: string
  label: string
  sub: string
  value: number
  err: number
  color: string
  human: boolean
  fit: DeviceFit | null
}

const rows = computed<Row[]>(() => {
  const list: Row[] = props.entries.map((r) => ({
    key: r.entrant_id,
    label: entrantShortLabel(r),
    sub: entrantDetail(r),
    value: r.metrics.quality.mean,
    err: r.metrics.quality.ci95,
    color: entrantColor(r),
    human: false,
    fit: props.runsHere?.[r.entrant_id] ?? null,
  }))
  if (props.human) {
    list.push({
      key: "human",
      label: props.human.label,
      sub: props.human.live ? "playing now" : "you",
      value: props.human.score,
      err: 0,
      color: HUMAN_COLOR,
      human: true,
      fit: null,
    })
  }
  // Stable for ties: the human only passes an entrant by strictly beating its mean.
  return list.sort((a, b) => b.value - a.value || (a.human ? 1 : b.human ? -1 : 0))
})

const scale = computed(() => Math.max(props.score.scaleMin, ...rows.value.map((r) => r.value + r.err)))
</script>

<template>
  <TransitionGroup tag="ol" name="rank" class="relative space-y-1">
    <li
      v-for="(row, i) in rows"
      :key="row.key"
      :data-selected="!row.human && selectedId === row.key"
      class="group relative flex items-center gap-3 rounded-lg px-3 py-2 transition-colors"
      :class="[
        row.human
          ? 'border border-queen-400/50 bg-queen-500/10 shadow-[0_0_24px_-8px_rgb(255_92_122/0.6)]'
          : selectedId === row.key
            ? 'bg-raised ring-1 ring-line-strong'
            : 'cursor-pointer hover:bg-raised/60',
        row.fit && !row.fit.ok ? 'opacity-55' : '',
      ]"
      :title="row.fit?.note"
      @click="!row.human && $emit('select', row.key)"
    >
      <span class="num w-5 text-right text-xs text-fg-subtle">{{ i + 1 }}</span>
      <span class="size-2 shrink-0 rounded-full" :style="{ background: row.color }" />
      <div class="min-w-0 flex-1">
        <p class="truncate text-sm" :class="row.human ? 'font-semibold text-queen-200' : 'text-fg'">{{ row.label }}</p>
        <p class="truncate font-mono text-[10px] text-fg-subtle">
          <template v-if="!row.human && selectedId === row.key"><span class="text-queen-300">● watching</span> · </template>{{ row.sub }}
        </p>
        <p v-if="row.fit && !row.fit.ok" data-cant-run class="truncate text-[10px] text-queen-300">✕ can't run here · {{ row.fit.note }}</p>
        <div class="mt-1 h-1 overflow-hidden rounded-full bg-sunken">
          <div
            class="h-full rounded-full transition-[width] duration-500"
            :style="{ width: `${(row.value / scale) * 100}%`, background: row.color }"
          />
        </div>
      </div>
      <p class="num w-12 text-right text-sm font-semibold" :class="row.human ? 'text-queen-200' : 'text-fg'">
        {{ score.compact(row.value) }}
      </p>
    </li>
  </TransitionGroup>
</template>

<style scoped>
.rank-move {
  transition: transform 0.55s cubic-bezier(0.22, 1, 0.36, 1);
}
.rank-enter-active {
  transition: all 0.4s ease;
}
.rank-enter-from {
  opacity: 0;
  transform: translateX(-12px);
}
</style>
