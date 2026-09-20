<script setup lang="ts">
// The players on offer, one card each. Game-agnostic; extra text (a trained player's record) goes in the slot.
defineProps<{ strategies: { id: string; label: string; description: string }[] }>()
</script>

<template>
  <section>
    <p class="eyebrow">The players</p>
    <div class="mt-3 grid gap-3 md:grid-cols-2">
      <div v-for="s in strategies" :key="s.id" class="card p-4">
        <p class="font-display font-semibold">{{ s.label }}</p>
        <p class="mt-1 text-sm text-fg-muted">{{ s.description }}</p>
      </div>
    </div>
    <div class="mt-3 max-w-3xl text-xs text-fg-subtle"><slot /></div>
  </section>
</template>
