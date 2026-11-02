<script setup lang="ts">
import type { CopyCatStages, CopyCatTarget, CopyCatTargetDetail } from "~/utils/copycat"

// One held-out image: what it was given, what a tracer and the pipeline made of it as the leaderboard scored them, and
// the clean SVG both are scored against. Then the pipeline again, here, in WebAssembly, with its stages.
const STRUCTURED = "baseline:structured"
const TRACER = "baseline:vtracer-nobg"
const SEVERITIES = ["clean", "light", "medium", "heavy"] as const

const api = useApi()
const targets = ref<CopyCatTarget[]>([])
const seed = ref<number | null>(null)
const detail = ref<CopyCatTargetDetail | null>(null)
const stages = ref<CopyCatStages | null>(null)
const failed = ref<string | null>(null)

onMounted(async () => {
  try {
    targets.value = await api.fetch<CopyCatTarget[]>("/games/copycat/targets")
    seed.value = targets.value.find((t) => t.severity === "medium")?.seed ?? targets.value[0]?.seed ?? null
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
})

watch(seed, async (s) => {
  if (s === null) return
  detail.value = null
  stages.value = null
  try {
    const loaded = await api.fetch<CopyCatTargetDetail>(`/games/copycat/targets/${s}`)
    if (s !== seed.value) return
    detail.value = loaded
    await loadCopyCat()
    const pixels = await dataUrlPixels(loaded.png)
    if (s === seed.value) stages.value = traceStages(pixels.data, pixels.width, pixels.height)
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
})

const note = (id: string) => {
  const scores = detail.value?.outputs[id]?.scores
  return scores ? `${scores.fidelity.toFixed(1)} dB · ${scores.shapes} shapes` : undefined
}
const cleanShapes = computed(() => detail.value?.clean.match(/<(rect|circle|ellipse|polygon|path)\b/g)?.length)
</script>

<template>
  <UiFigure title="One image, two tracers, and the answer" kind="Live">
    <p v-if="failed" class="text-sm text-fg-subtle">The held-out images couldn't be loaded ({{ failed }}).</p>
    <div v-else class="space-y-5">
      <div class="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <CopyCatImage label="Given" :src="detail?.png" :note="detail?.severity" />
        <CopyCatImage label="VTracer" :svg="detail?.outputs[TRACER]?.svg" :note="note(TRACER)">
          <template #empty>Loading…</template>
        </CopyCatImage>
        <CopyCatImage label="The pipeline" :svg="detail?.outputs[STRUCTURED]?.svg" :note="note(STRUCTURED)">
          <template #empty>Loading…</template>
        </CopyCatImage>
        <CopyCatImage label="Clean" :svg="detail?.clean" :note="cleanShapes ? `${cleanShapes} shapes` : undefined" />
      </div>
      <div class="space-y-2">
        <div v-for="severity in SEVERITIES" :key="severity" class="flex items-center gap-3">
          <span class="label w-16 shrink-0">{{ severity }}</span>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="t in targets.filter((x) => x.severity === severity)"
              :key="t.seed"
              type="button"
              class="size-10 overflow-hidden rounded-[6px] border transition"
              :class="t.seed === seed ? 'border-queen-400 ring-1 ring-queen-400' : 'border-line hover:border-line-strong'"
              :aria-label="`Image ${t.seed}, ${severity}`"
              @click="seed = t.seed"
            >
              <img :src="t.png" alt="" class="size-full" style="image-rendering: pixelated" />
            </button>
          </div>
        </div>
      </div>
      <div>
        <p class="label mb-3">The pipeline again, running in your browser</p>
        <CopyCatStages v-if="stages" :stages="stages" />
        <UiEmpty v-else compact class="h-40">Tracing…</UiEmpty>
      </div>
    </div>
    <template #caption>
      These are held-out images: the pipeline was never tuned on them. Pull the layers apart to see a hidden part come back whole.
    </template>
  </UiFigure>
</template>
