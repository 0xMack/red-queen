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
    <button v-if="canPause" class="btn-ghost btn-sm w-24" :aria-pressed="paused" @click="emit('update:paused', !props.paused)">
      <svg viewBox="0 0 10 10" class="size-2.5" fill="currentColor" aria-hidden="true">
        <path v-if="paused" d="M2 1l7 4-7 4z" />
        <path v-else d="M2 1h2v8H2zM6 1h2v8H6z" />
      </svg>
      {{ paused ? "Resume" : "Pause" }}
    </button>
    <UiSegmented
      v-if="showSpeed"
      :model-value="speed"
      :options="speeds.map((s) => ({ value: s, label: speedLabel(s) }))"
      aria-label="Speed"
      @update:model-value="(s) => emit('update:speed', s)"
    />
    <button class="btn-primary btn-sm ml-auto" @click="emit('newGame')">↻ {{ newGameLabel }}</button>
    <slot />
  </div>
</template>
