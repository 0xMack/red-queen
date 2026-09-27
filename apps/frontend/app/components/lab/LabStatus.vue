<script setup lang="ts">
// What a lab says when a run ends (the default slot: what to try next) or fails.
defineProps<{ status: string; error?: string | null }>()
</script>

<template>
  <p v-if="error" class="mt-3 text-xs text-queen-300">{{ error }}</p>
  <p v-else-if="status === 'done' && $slots.default" class="mt-3 flex items-center gap-2 text-xs text-life-300">
    <span class="size-1.5 rotate-45 bg-life-400" /><slot />
  </p>
</template>
