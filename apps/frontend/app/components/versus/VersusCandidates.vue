<script setup lang="ts">
import type { Candidates } from "~/types/versus"

// Every legal move with the score its strategy gave it, the chosen one marked. Game-agnostic (any bot that
// exposes `scores()`; the engine wrote the move notation). Just the list: the player panel around it
// (VersusPlayerPanel) supplies the header, and gives this a fixed height -- it scrolls -- so the layout
// doesn't move with the number of legal moves.
const props = defineProps<{ candidates: Candidates | null }>()

const rows = computed(() => {
  const c = props.candidates
  if (!c) return []
  const best = Math.max(...c.scores)
  const worst = Math.min(...c.scores)
  const span = best - worst
  return c.labels.map((label, i) => ({
    label,
    score: c.scores[i]!,
    // Bars are relative within this decision (scores from different strategies aren't on one scale).
    width: span > 0 ? 8 + ((c.scores[i]! - worst) / span) * 92 : 100,
    best: c.scores[i] === best,
    chosen: c.chosen === i,
  }))
})
const flat = computed(() => rows.value.length > 0 && rows.value.every((r) => r.score === rows.value[0]!.score))
</script>

<template>
  <div class="flex h-full min-h-0 flex-col">
    <p v-if="!candidates" class="text-xs text-fg-subtle">
      Shown once this seat is played by a bot that scores its moves (random and first-legal don't; humans don't).
    </p>
    <template v-else>
      <ol class="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        <li v-for="(row, i) in rows" :key="i" class="flex items-center gap-2 text-xs" :class="row.chosen ? 'text-fg' : 'text-fg-muted'">
          <span class="w-16 shrink-0 truncate font-mono">{{ row.label }}</span>
          <span class="relative h-2 flex-1 overflow-hidden rounded-full bg-raised">
            <span
              class="absolute inset-y-0 left-0 rounded-full transition-[width] duration-300"
              :class="row.chosen ? 'bg-queen-400' : row.best ? 'bg-life-400/70' : 'bg-line-strong'"
              :style="{ width: `${row.width}%` }"
            />
          </span>
          <span class="num w-12 shrink-0 text-right">{{ Math.abs(row.score) >= 500 ? (row.score > 0 ? "win" : "loss") : row.score.toFixed(2) }}</span>
          <span class="w-3 shrink-0 text-queen-300">{{ row.chosen ? "◀" : "" }}</span>
        </li>
      </ol>
      <p v-if="flat" class="mt-2 shrink-0 text-[11px] text-fg-subtle">Every move scores the same here, so the pick is a tie-break.</p>
    </template>
  </div>
</template>
