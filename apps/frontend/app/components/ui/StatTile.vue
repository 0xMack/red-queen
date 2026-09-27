<script setup lang="ts">
// One headline number: a label, the value large, a hint underneath. For a row of small readouts use `UiStats`.
defineProps<{ label: string; value: string | number; hint?: string; tone?: "default" | "queen" | "life" | "gold" }>()
</script>

<template>
  <div class="card px-4 py-3.5">
    <p class="label truncate">{{ label }}</p>
    <p
      class="num mt-1.5 truncate text-2xl leading-none font-medium tracking-tight"
      :class="{
        'text-fg': !tone || tone === 'default',
        'text-queen-300': tone === 'queen',
        'text-life-300': tone === 'life',
        'text-gold-300': tone === 'gold',
      }"
    >
      {{ value }}
    </p>
    <p v-if="hint || $slots.hint" class="mt-1.5 truncate text-xs text-fg-subtle"><slot name="hint">{{ hint }}</slot></p>
    <slot />
  </div>
</template>
