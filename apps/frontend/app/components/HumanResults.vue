<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"
import type { HumanHistory } from "~/utils/leaderboard"

// After a human game: the score counts up, then one bar per algorithm fills in (staggered) with how
// often that score beats it -- across the algorithm's own 200 held-out games, not just its mean
// (utils/leaderboard.ts beatRate). The ranking itself animates in the side leaderboard.
const props = defineProps<{ entries: EvaluationRecord[]; score: number; history: HumanHistory; newBest: boolean }>()
defineEmits<{ again: []; watch: [] }>()

const shown = ref(0)
const reveal = ref(false)
let frame = 0
onMounted(() => {
  const started = performance.now()
  const duration = Math.min(1200, 250 + props.score * 60)
  const tick = (now: number) => {
    const t = Math.min(1, (now - started) / duration)
    shown.value = Math.round(props.score * (1 - (1 - t) ** 3))
    if (t < 1) frame = requestAnimationFrame(tick)
    else reveal.value = true
  }
  frame = requestAnimationFrame(tick)
})
onUnmounted(() => cancelAnimationFrame(frame))

const rows = computed(() =>
  [...props.entries]
    .sort((a, b) => b.metrics.quality.mean - a.metrics.quality.mean)
    .map((r) => ({ id: r.entrant_id, label: entrantShortLabel(r), sub: entrantDetail(r), color: entrantColor(r), mean: r.metrics.quality.mean, rate: beatRate(props.score, r) })),
)
const rank = computed(() => rows.value.filter((r) => r.mean > props.score).length + 1)
const beatenOnAverage = computed(() => rows.value.filter((r) => props.score > r.mean))
const headline = computed(() => {
  const above = rows.value.find((r, i) => r.mean > props.score && (rows.value[i + 1]?.mean ?? -1) <= props.score)
  const below = beatenOnAverage.value[0]
  if (!below) return `Every algorithm averages more than ${props.score}. Try again?`
  if (!above) return "Better than every algorithm's average — including the best one."
  return `Between ${above.label} (${above.mean.toFixed(1)}) and ${below.label} (${below.mean.toFixed(1)}).`
})
const average = computed(() =>
  props.history.games.length ? props.history.games.reduce((a, b) => a + b, 0) / props.history.games.length : 0,
)
</script>

<template>
  <div class="rounded-xl border border-queen-400/30 bg-sunken p-5">
    <div class="flex items-start justify-between gap-4">
      <div>
        <p class="text-[11px] tracking-wide text-fg-subtle uppercase">Your score</p>
        <p class="num font-display text-6xl leading-none font-bold text-queen-200">{{ shown }}</p>
      </div>
      <div class="text-right">
        <Transition name="pop">
          <span v-if="reveal && newBest" class="inline-block rounded-full bg-gold-400/15 px-2.5 py-1 text-xs font-semibold text-gold-300">★ New personal best</span>
        </Transition>
        <p class="mt-2 text-xs text-fg-subtle">
          best <span class="num text-fg">{{ history.best }}</span> · avg
          <span class="num text-fg">{{ average.toFixed(1) }}</span> · {{ history.games.length }} games
        </p>
      </div>
    </div>

    <Transition name="fade-up">
      <div v-if="reveal" class="mt-4">
        <p class="font-display text-lg font-semibold">
          You'd rank <span class="text-queen-200">#{{ rank }}</span> of {{ rows.length + 1 }}
        </p>
        <p class="text-sm text-fg-muted">{{ headline }}</p>

        <p class="mt-4 text-[11px] tracking-wide text-fg-subtle uppercase">This score beats…</p>
        <ul class="mt-2 space-y-2">
          <li v-for="(row, i) in rows" :key="row.id" class="text-xs">
            <div class="flex items-center justify-between gap-2">
              <span class="flex min-w-0 items-center gap-1.5 truncate text-fg-muted">
                <span class="size-1.5 shrink-0 rounded-full" :style="{ background: row.color }" />
                <span class="truncate">{{ row.label }}</span>
                <span class="font-mono text-[10px] text-fg-subtle">{{ row.sub }}</span>
              </span>
              <span class="num shrink-0" :class="row.rate >= 0.5 ? 'text-queen-200' : 'text-fg-subtle'">
                {{ Math.round(row.rate * 100) }}% of its games
              </span>
            </div>
            <div class="mt-1 h-1.5 overflow-hidden rounded-full bg-raised">
              <div
                class="bar h-full rounded-full"
                :style="{ '--w': `${row.rate * 100}%`, background: row.rate >= 0.5 ? '#ff5c7a' : row.color, animationDelay: `${i * 90}ms` }"
              />
            </div>
          </li>
        </ul>

        <div class="mt-5 flex flex-wrap gap-2">
          <button class="btn-primary" @click="$emit('again')">Play again <span class="text-xs opacity-70">(space)</span></button>
          <button class="btn-ghost" @click="$emit('watch')">Back to watching</button>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.bar {
  width: 0;
  animation: fill 0.8s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
@keyframes fill {
  to {
    width: var(--w);
  }
}
.fade-up-enter-active,
.pop-enter-active {
  transition: all 0.4s ease;
}
.fade-up-enter-from {
  opacity: 0;
  transform: translateY(8px);
}
.pop-enter-from {
  opacity: 0;
  transform: scale(0.6);
}
</style>
