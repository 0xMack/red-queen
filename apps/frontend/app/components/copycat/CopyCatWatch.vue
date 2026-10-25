<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { CopyCatStages, CopyCatTarget, CopyCatTargetDetail } from "~/utils/copycat"

// Copy Cat's Watch stage (docs/design/0022): pick a held-out image and see what the entrant on stage made of it, beside
// the corrupted input it was given and the clean SVG it was scored against, with its scores for that image. For the
// structured pipeline the stage also re-runs it here, in WebAssembly, to show its stages -- and checks it wrote the
// same SVG the leaderboard scored.
// Props spelled out (StageProps, games/types.ts): the SFC compiler can't resolve an imported type as the whole props.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()

const api = useApi()
const route = useRoute()
const router = useRouter()

const targets = ref<CopyCatTarget[]>([])
const failed = ref<string | null>(null)
const details = reactive(new Map<number, CopyCatTargetDetail>())
const SEVERITIES = ["clean", "light", "medium", "heavy"] as const

const seed = computed<number | null>(() => {
  const asked = Number(route.query.image)
  if (targets.value.some((t) => t.seed === asked)) return asked
  return targets.value.find((t) => t.severity === "medium")?.seed ?? targets.value[0]?.seed ?? null
})
const pick = (s: number) => router.replace({ query: { ...route.query, image: String(s) } })

onMounted(async () => {
  try {
    targets.value = await api.fetch<CopyCatTarget[]>("/games/copycat/targets")
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
})
watch(
  seed,
  async (s) => {
    if (s === null || details.has(s)) return
    try {
      details.set(s, await api.fetch<CopyCatTargetDetail>(`/games/copycat/targets/${s}`))
    } catch (e) {
      failed.value = e instanceof Error ? e.message : String(e)
    }
  },
  { immediate: true },
)

const detail = computed(() => (seed.value === null ? null : (details.get(seed.value) ?? null)))
const output = computed(() => (props.entry && detail.value ? (detail.value.outputs[props.entry.entrant_id] ?? null) : null))
const clean = computed(() => (detail.value ? describeSvgSafe(detail.value.clean) : null))

// the pipeline, re-run here when it is the entrant on stage
const isPipeline = computed(() => props.entry?.entrant_id === "baseline:structured")
const stages = ref<CopyCatStages | null>(null)
watch(
  [isPipeline, detail],
  async ([pipeline, d]) => {
    stages.value = null
    if (!pipeline || !d) return
    await loadCopyCat()
    const pixels = await dataUrlPixels(d.png)
    if (detail.value === d) stages.value = traceStages(pixels.data, pixels.width, pixels.height)
  },
  { immediate: true },
)
const sameAsScored = computed(() => (stages.value && output.value ? stages.value.svg === output.value.svg : null))

function describeSvgSafe(svg: string) {
  const counts = svg.match(/<(rect|circle|ellipse|polygon|path)\b/g)?.length ?? 0
  const colours = new Set(svg.match(/fill="#[0-9a-f]{6}"/g) ?? []).size
  return { shapes: counts, palette: colours }
}

const scoreStats = computed(() => {
  const s = output.value?.scores
  if (!s) return []
  return [
    { label: "Clean fidelity", value: `${s.fidelity.toFixed(1)} dB`, tone: "queen" as const },
    { label: "Alpha IoU", value: s.alpha_iou.toFixed(2) },
    { label: "Layer recovery", value: s.layer_recovery.toFixed(2) },
    { label: "Shapes", value: `${s.shapes} (clean ${clean.value?.shapes ?? "?"})` },
    { label: "Colours", value: `${s.palette} (clean ${clean.value?.palette ?? "?"})` },
    { label: "Size", value: `${(s.bytes / 1024).toFixed(1)} KB` },
  ]
})
const severityNote = (t: CopyCatTarget) => {
  const background = (t.levels.background as { kind?: string } | undefined)?.kind
  return [t.severity, background && t.severity !== "clean" ? `${background} background` : null].filter(Boolean).join(" · ")
}
</script>

<template>
  <div class="space-y-5">
    <p v-if="failed" class="text-sm text-fg-subtle">The held-out images couldn't be loaded ({{ failed }}).</p>

    <div class="grid gap-4 sm:grid-cols-3">
      <CopyCatImage label="The image it was given" :src="detail?.png" :note="detail ? severityNote(detail) : undefined" />
      <CopyCatImage :label="entry ? `${entry.label}'s SVG` : 'An entrant\'s SVG'" :svg="output?.svg" :note="output ? `${output.scores.shapes} shapes` : undefined">
        <template #empty>{{ entry && detail ? "No SVG recorded for this image" : "Loading…" }}</template>
      </CopyCatImage>
      <CopyCatImage label="The clean SVG behind it" :svg="detail?.clean" :note="clean ? `${clean.shapes} shapes` : undefined" />
    </div>

    <UiStats v-if="scoreStats.length" :items="scoreStats" :cols="6" />
    <p v-if="output?.scores.refused" class="text-xs text-queen-300">Refused: {{ output.scores.refused }} -- scored as an empty SVG.</p>

    <div class="space-y-2">
      <div v-for="severity in SEVERITIES" :key="severity" class="flex items-center gap-3">
        <span class="label w-16 shrink-0">{{ severity }}</span>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="t in targets.filter((x) => x.severity === severity)"
            :key="t.seed"
            type="button"
            class="size-11 overflow-hidden rounded-[6px] border transition"
            :class="t.seed === seed ? 'border-queen-400 ring-1 ring-queen-400' : 'border-line hover:border-line-strong'"
            :title="`Image ${t.seed} (${severityNote(t)})`"
            @click="pick(t.seed)"
          >
            <img :src="t.png" :alt="`held-out image ${t.seed}`" class="size-full" style="image-rendering: pixelated" />
          </button>
        </div>
      </div>
    </div>

    <UiPanel v-if="isPipeline" label="Live" title="The pipeline, run again in your browser">
      <template #actions>
        <UiBadge v-if="sameAsScored === true" tone="life" dot>same SVG as the leaderboard scored</UiBadge>
        <UiBadge v-else-if="sameAsScored === false" tone="gold" dot>differs from the scored SVG</UiBadge>
      </template>
      <CopyCatStages v-if="stages" :stages="stages" />
      <UiEmpty v-else compact class="h-40">Tracing…</UiEmpty>
    </UiPanel>
  </div>
</template>
