<script setup lang="ts">
import type { RenderState } from "~/types/games"

// A lab's stage for a grid game: the current policy playing, or a placeholder (keeping the board's footprint) until
// training starts. The caption slot is for what the policy is choosing between (`UiBars`).
withDefaults(defineProps<{ board: RenderState | null; live: boolean | null; placeholder: string; tickMs?: number }>(), { tickMs: 90 })
</script>

<template>
  <div>
    <div class="mx-auto w-full max-w-[340px]">
      <GridBoard v-if="board" :state="board" :tick-ms="tickMs" />
      <UiEmpty v-else square>{{ live === null ? "Checking this device…" : placeholder }}</UiEmpty>
    </div>
    <div class="mx-auto mt-3 max-w-[340px] text-xs text-fg-subtle">
      <slot />
    </div>
  </div>
</template>
