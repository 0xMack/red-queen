<script setup lang="ts">
// Controls for a lab that runs a generation at a time on the main thread (the evolution demos): play/pause, step
// one or ten generations, start over from a new seed.
defineProps<{ playing: boolean }>()
const emit = defineEmits<{ toggle: []; step: [n: number]; reset: [] }>()
</script>

<template>
  <div class="flex flex-wrap items-center gap-2">
    <button :class="playing ? 'btn-ghost btn-sm' : 'btn-accent btn-sm'" class="w-20" @click="emit('toggle')">{{ playing ? "Pause" : "Play" }}</button>
    <button class="btn-ghost btn-sm" @click="emit('step', 1)">Step</button>
    <button class="btn-ghost btn-sm" @click="emit('step', 10)">+10</button>
    <button class="btn-quiet btn-sm" @click="emit('reset')">New start</button>
    <slot />
  </div>
</template>
