<script setup lang="ts">
// A small status pill. `dot` adds a leading dot; `pulse` makes it breathe (something live).
withDefaults(defineProps<{ tone?: "neutral" | "queen" | "life" | "signal" | "gold"; dot?: boolean; pulse?: boolean }>(), { tone: "neutral" })
const TONE = {
  neutral: "border-line-strong text-fg-muted",
  queen: "border-queen-400/40 bg-queen-500/10 text-queen-300",
  life: "border-life-400/35 bg-life-400/10 text-life-300",
  signal: "border-signal-400/35 bg-signal-400/10 text-signal-300",
  gold: "border-gold-400/35 bg-gold-400/10 text-gold-300",
} as const
</script>

<template>
  <span class="inline-flex items-center gap-1.5 rounded-full border px-2 py-px font-mono text-[10.5px] whitespace-nowrap" :class="TONE[tone]">
    <span v-if="dot || pulse" class="size-1.5 rounded-full bg-current" :class="pulse ? 'animate-live-pulse' : ''" />
    <slot />
  </span>
</template>
