<script setup lang="ts">
defineProps<{ score: number; step: number; reward: number }>()
</script>

<template>
  <div class="flex items-center gap-6 text-sm">
    <span>score: <span class="font-mono">{{ score }}</span></span>
    <span>step: <span class="font-mono">{{ step }}</span></span>
    <span>reward: <span class="font-mono">{{ reward.toFixed(2) }}</span></span>
  </div>
</template>
