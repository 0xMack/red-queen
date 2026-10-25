<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { CopyCatStages } from "~/utils/copycat"

// Copy Cat's Play stage (docs/design/0022): drop in your own image -- a generated logo or icon -- and the structured
// pipeline turns it into a layered, transparent SVG here, in WebAssembly. Nothing is uploaded: the file is read, scaled
// to at most TRACE_MAX_SIDE px and traced on this device, and the SVG can be downloaded.
// Props spelled out (StageProps, games/types.ts): the SFC compiler can't resolve an imported type as the whole props.
defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()

const input = ref<string | null>(null)
const name = ref("traced")
const stages = ref<CopyCatStages | null>(null)
const busy = ref(false)
const failed = ref<string | null>(null)
const dragging = ref(false)
const picker = ref<HTMLInputElement | null>(null)

async function trace(file: File) {
  if (!file.type.startsWith("image/")) {
    failed.value = `${file.name} isn't an image`
    return
  }
  failed.value = null
  busy.value = true
  stages.value = null
  name.value = file.name.replace(/\.[^.]+$/, "") || "traced"
  try {
    const [pixels] = await Promise.all([imageFilePixels(file), loadCopyCat()])
    const canvas = document.createElement("canvas")
    canvas.width = pixels.width
    canvas.height = pixels.height
    canvas.getContext("2d")!.putImageData(pixels, 0, 0)
    input.value = canvas.toDataURL("image/png")
    await nextTick()
    await new Promise((resolve) => setTimeout(resolve, 20)) // let "Tracing…" paint before the main thread is busy
    stages.value = traceStages(pixels.data, pixels.width, pixels.height)
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

function onDrop(event: DragEvent) {
  dragging.value = false
  const file = event.dataTransfer?.files[0]
  if (file) trace(file)
}
function onPick(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) trace(file)
}

function download() {
  if (!stages.value) return
  const url = URL.createObjectURL(new Blob([stages.value.svg], { type: "image/svg+xml" }))
  const a = document.createElement("a")
  a.href = url
  a.download = `${name.value}.svg`
  a.click()
  URL.revokeObjectURL(url)
}
const size = computed(() => (stages.value ? `${(new Blob([stages.value.svg]).size / 1024).toFixed(1)} KB` : undefined))
</script>

<template>
  <div class="space-y-5">
    <div
      class="flex flex-col items-center justify-center gap-2 rounded-[10px] border border-dashed px-4 py-8 text-center transition"
      :class="dragging ? 'border-queen-400 bg-queen-500/10' : 'border-line-strong bg-sunken'"
      @dragover.prevent="dragging = true"
      @dragleave="dragging = false"
      @drop.prevent="onDrop"
    >
      <p class="text-sm text-fg">Drop an image here -- a generated logo or icon works best</p>
      <p class="text-xs text-fg-subtle">
        It is traced on this device, at up to {{ TRACE_MAX_SIDE }} px a side. Nothing is uploaded.
      </p>
      <button type="button" class="btn-ghost btn-sm mt-1" @click="picker?.click()">Choose an image</button>
      <input ref="picker" type="file" accept="image/*" class="hidden" @change="onPick" />
    </div>
    <p v-if="failed" class="text-sm text-queen-300">{{ failed }}</p>

    <div v-if="input" class="grid gap-4 sm:grid-cols-2">
      <CopyCatImage label="Your image" :src="input" :note="stages ? `${stages.width}×${stages.height}` : undefined" />
      <CopyCatImage label="The SVG it recovered" :svg="stages?.svg" :note="size">
        <template #empty>{{ busy ? "Tracing…" : "Nothing yet" }}</template>
      </CopyCatImage>
    </div>
    <div v-if="stages" class="flex flex-wrap items-center gap-3">
      <button type="button" class="btn-primary btn-sm" @click="download">Download {{ name }}.svg</button>
      <span class="text-xs text-fg-subtle">One &lt;g&gt; per part, bottom first, with a transparent background.</span>
    </div>
    <UiPanel v-if="stages" label="Stages" title="How it got there">
      <CopyCatStages :stages="stages" />
    </UiPanel>
  </div>
</template>
