<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"

// Compact, rank-ordered leaderboard for the game page's side panel. Doubles as the "what am I
// watching" selector (click an entrant to play it), and -- when `human` is set -- slots the
// player in by score, animating their row up the ranking as they play (TransitionGroup moves rows
// with a FLIP transition whenever the order changes).
const props = defineProps<{
  entries: EvaluationRecord[]
  selectedId?: string | null
  human?: { score: number; label: string; live: boolean } | null
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
    })
  }
  // Stable for ties: the human only passes an entrant by strictly beating its mean.
  return list.sort((a, b) => b.value - a.value || (a.human ? 1 : b.human ? -1 : 0))
})

const scale = computed(() => Math.max(1, ...rows.value.map((r) => r.value + r.err)))
</script>

<template>
  <TransitionGroup tag="ol" name="rank" class="relative space-y-1">
    <li
      v-for="(row, i) in rows"
      :key="row.key"
      class="group relative flex items-center gap-3 rounded-lg px-3 py-2 transition-colors"
      :class="[
        row.human
          ? 'border border-queen-400/50 bg-queen-500/10 shadow-[0_0_24px_-8px_rgb(255_92_122/0.6)]'
          : selectedId === row.key
            ? 'bg-raised ring-1 ring-line-strong'
            : 'cursor-pointer hover:bg-raised/60',
      ]"
      @click="!row.human && $emit('select', row.key)"
    >
      <span class="num w-5 text-right text-xs text-fg-subtle">{{ i + 1 }}</span>
      <span class="size-2 shrink-0 rounded-full" :style="{ background: row.color }" />
      <div class="min-w-0 flex-1">
        <p class="truncate text-sm" :class="row.human ? 'font-semibold text-queen-200' : 'text-fg'">{{ row.label }}</p>
        <p class="truncate font-mono text-[10px] text-fg-subtle">
          <template v-if="!row.human && selectedId === row.key"><span class="text-queen-300">● watching</span> · </template>{{ row.sub }}
        </p>
        <div class="mt-1 h-1 overflow-hidden rounded-full bg-sunken">
          <div
            class="h-full rounded-full transition-[width] duration-500"
            :style="{ width: `${(row.value / scale) * 100}%`, background: row.color }"
          />
        </div>
      </div>
      <p class="num w-10 text-right text-sm font-semibold" :class="row.human ? 'text-queen-200' : 'text-fg'">
        {{ row.human ? row.value : row.value.toFixed(1) }}
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
