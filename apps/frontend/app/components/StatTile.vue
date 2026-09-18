<script setup lang="ts">
defineProps<{ label: string; value: string | number; hint?: string; tone?: "default" | "queen" | "life" | "gold" }>()
</script>

<template>
  <div class="card px-4 py-3">
    <p class="text-[11px] font-medium tracking-wide text-fg-subtle uppercase">{{ label }}</p>
    <p
      class="num mt-1 truncate text-xl font-semibold"
      :class="{
        'text-fg': !tone || tone === 'default',
        'text-queen-300': tone === 'queen',
        'text-life-400': tone === 'life',
        'text-gold-300': tone === 'gold',
      }"
    >
      {{ value }}
    </p>
    <p v-if="hint || $slots.hint" class="mt-0.5 truncate text-xs text-fg-subtle"><slot name="hint">{{ hint }}</slot></p>
    <slot />
  </div>
</template>
