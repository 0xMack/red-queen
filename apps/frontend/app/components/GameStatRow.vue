<script setup lang="ts">
defineProps<{ score: number; step: number; reward: number }>()
</script>

<template>
  <div class="flex items-center gap-5 text-xs text-fg-subtle">
    <span>score <span class="num ml-1 text-sm text-life-300">{{ score }}</span></span>
    <span>step <span class="num ml-1 text-sm text-fg">{{ step }}</span></span>
    <span>reward <span class="num ml-1 text-sm text-fg">{{ reward.toFixed(2) }}</span></span>
  </div>
</template>
