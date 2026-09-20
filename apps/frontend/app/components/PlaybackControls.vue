<script setup lang="ts">
import { SPEEDS, speedLabel } from "~/utils/playback"

// The playback controls every game shows under its board: pause / resume, a speed (a multiplier, see
// utils/playback.ts), and start a new game. Presentational and game-agnostic: the owner (a versus session, a
// Snake worker session) holds the state and does the work. `canPause` / `showSpeed` are off where a game has
// nothing to pause or nothing to speed up (a human-only game). The default slot is for anything extra that
// belongs on the same row.
const props = withDefaults(
  defineProps<{
    paused?: boolean
    speed: number
    speeds?: readonly number[]
    canPause?: boolean
    showSpeed?: boolean
    newGameLabel?: string
  }>(),
  { paused: false, speeds: () => SPEEDS, canPause: true, showSpeed: true, newGameLabel: "New game" },
)
const emit = defineEmits<{ newGame: []; "update:paused": [paused: boolean]; "update:speed": [speed: number] }>()
</script>

<template>
  <div class="flex flex-wrap items-center gap-2" role="group" aria-label="Playback controls" data-testid="playback">
    <button v-if="canPause" class="btn-ghost btn-sm w-28 justify-center" :aria-pressed="paused" @click="emit('update:paused', !props.paused)">
      {{ paused ? "▶ Resume" : "⏸ Pause" }}
    </button>
    <div v-if="showSpeed" class="flex items-center gap-2">
      <span class="text-[11px] text-fg-subtle">speed</span>
      <div class="flex rounded-lg border border-line bg-sunken p-0.5" role="radiogroup" aria-label="Speed">
        <button
          v-for="s in speeds"
          :key="s"
          class="num rounded-md px-2.5 py-1 text-xs transition"
          :class="speed === s ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
          role="radio"
          :aria-checked="speed === s"
          @click="emit('update:speed', s)"
        >
          {{ speedLabel(s) }}
        </button>
      </div>
    </div>
    <button class="btn-primary btn-sm ml-auto" @click="emit('newGame')">↻ {{ newGameLabel }}</button>
    <slot />
  </div>
</template>
