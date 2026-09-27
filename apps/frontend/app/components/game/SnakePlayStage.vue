<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"

// Snake's Play stage for the shared game page: a human steering the same board and game the algorithms
// play (HumanPlay). Reports the score up so the side leaderboard can slot "You" in; a finished game
// also updates this browser's personal best.
// Props are spelled out (the shape of `StageProps` in games/types.ts): the SFC compiler can't resolve an
// imported type used as the whole props type without TypeScript installed.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()
const emit = defineEmits<{ score: [score: number, live: boolean, label?: string]; exit: [] }>()

function onScore(score: number, live: boolean) {
  if (live) return emit("score", score, true)
  emit("score", score, false, score >= loadHumanHistory("snake").best ? "You (best)" : "You")
}
</script>

<template>
  <HumanPlay game="snake" :entries="props.entries" @score="onScore" @exit="emit('exit')" />
</template>
