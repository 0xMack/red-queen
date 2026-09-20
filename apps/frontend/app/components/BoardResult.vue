<script setup lang="ts">
// The result laid over a game's board when a game ends -- the same overlay for every game (Checkers: "Red wins" /
// "Draw"; Snake: "Game over" with its score), so a finished game looks like a finished game wherever you are.
// Place it inside a `relative` wrapper around the board. It waits for whoever owns it to say the board is *still*
// (a board mid-animation with a result on top reads as a game that keeps going). A plain v-if + CSS animation, not
// <Transition>: a game can end every couple of seconds at full speed, and Transition's enter/leave bookkeeping
// racing that toggle is exactly the kind of thing that has thrown from Vue's patcher before.
defineProps<{
  title: string
  sub?: string
  tone?: "red" | "black" | "draw"
}>()
</script>

<template>
  <div class="result pointer-events-none absolute inset-0 flex items-center justify-center" role="status">
    <div class="rounded-2xl border border-white/10 bg-black/60 px-7 py-4 text-center backdrop-blur-sm">
      <p class="font-display text-2xl font-semibold" :class="tone === 'red' ? 'text-queen-300' : tone === 'black' ? 'text-fg' : 'text-gold-300'">{{ title }}</p>
      <p v-if="sub" class="mt-1 text-xs text-fg-muted">{{ sub }}</p>
    </div>
  </div>
</template>

<style scoped>
.result {
  animation: result-in 350ms cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
}
@keyframes result-in {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
@media (prefers-reduced-motion: reduce) {
  .result {
    animation: none;
  }
}
</style>
