<script setup lang="ts">
import { sketchStart, step, type SnakeSketch } from "~/data/explainers/snakeSketch"

// An observer watching a moving snake: the representation panel's live view. A scripted snake (the explainers'
// sketch, not the Rust game) steps a few times a second, and the observer's picture is redrawn from each position.
defineProps<{ id: string }>()

const state = shallowRef<SnakeSketch>(sketchStart())
const playing = ref(true)
let timer: ReturnType<typeof setInterval> | null = null
function run() {
  if (timer) clearInterval(timer)
  timer = setInterval(() => playing.value && (state.value = step(state.value)), 420)
}
onMounted(run)
onUnmounted(() => timer && clearInterval(timer))
</script>

<template>
  <div>
    <div class="well overflow-hidden p-1.5"><ObserverGlyph :id="id" :state="state" /></div>
    <div class="mt-2 flex items-center gap-2">
      <button type="button" class="btn-ghost btn-sm" @click="playing = !playing">{{ playing ? "Pause" : "Play" }}</button>
      <button type="button" class="btn-quiet btn-sm" :disabled="playing" @click="state = step(state)">Step</button>
      <button type="button" class="btn-quiet btn-sm" @click="state = sketchStart()">Reset</button>
      <span class="ml-auto text-[11px] text-fg-subtle">a scripted snake, for illustration</span>
    </div>
  </div>
</template>
