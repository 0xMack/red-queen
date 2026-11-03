<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { CopyCatStages, CopyCatTarget, CopyCatTargetDetail } from "~/utils/copycat"

// Copy Cat's Watch stage (docs/design/0022): pick a held-out image and see what the entrant on stage made of it -- its
// timelapse, ending on the SVG the leaderboard scored (CopyCatFilm) -- beside the corrupted input it was given and the
// clean SVG it was scored against. For the structured pipeline the stage also re-runs it here, in WebAssembly, to show
// its stages -- and checks it wrote the same SVG the leaderboard scored.
// Props spelled out (StageProps, games/types.ts): the SFC compiler can't resolve an imported type as the whole props.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()

const api = useApi()
const route = useRoute()
const router = useRouter()

const targets = ref<CopyCatTarget[]>([])
const failed = ref<string | null>(null)
const details = reactive(new Map<string, CopyCatTargetDetail>())
const SEVERITIES = ["clean", "light", "medium", "heavy"] as const

// an image is `set/seed` (seeds repeat across sets); a bare seed is a link from before there was more than one set
const id = computed<string | null>(() => {
  const raw = String(route.query.image ?? "")
  const asked = raw.includes("/") ? raw : `synthetic/${raw}`
  if (targets.value.some((t) => t.id === asked)) return asked
  return targets.value.find((t) => t.severity === "medium")?.id ?? targets.value[0]?.id ?? null
})
const pick = (next: string) => router.replace({ query: { ...route.query, image: next } })

const sets = computed(() => [...new Set(targets.value.map((t) => t.set))])
const shownSet = computed({
  get: () => id.value?.split("/")[0] ?? sets.value[0] ?? "",
  set: (s: string) => {
    const first = targets.value.find((t) => t.set === s && t.severity === "medium") ?? targets.value.find((t) => t.set === s)
    if (first) pick(first.id)
  },
})
const setOptions = computed(() =>
  sets.value.map((s) => ({ value: s, label: `${s} (${targets.value.filter((t) => t.set === s).length})` })),
)

onMounted(async () => {
  try {
    targets.value = await api.fetch<CopyCatTarget[]>("/games/copycat/targets")
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
})
watch(
  id,
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

const detail = computed(() => (id.value === null ? null : (details.get(id.value) ?? null)))
const output = computed(() => (props.entry && detail.value ? (detail.value.outputs[props.entry.entrant_id] ?? null) : null))

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

const severityNote = (t: CopyCatTarget) => {
  const background = (t.levels.background as { kind?: string } | undefined)?.kind
  return [t.severity, background && t.severity !== "clean" ? `${background} background` : null].filter(Boolean).join(" · ")
}
</script>

<template>
  <div class="space-y-5">
    <p v-if="failed" class="text-sm text-fg-subtle">The held-out images couldn't be loaded ({{ failed }}).</p>

    <CopyCatFilm
      :entrant-id="entry?.entrant_id ?? null"
      :entrant-label="entry?.label ?? 'This entrant'"
      :detail="detail"
      :result="output"
      :given-note="detail ? severityNote(detail) : undefined"
    />

    <div class="space-y-2">
      <UiSegmented v-if="sets.length > 1" v-model="shownSet" :options="setOptions" aria-label="Which held-out set" />
      <div v-for="severity in SEVERITIES" :key="severity" class="flex items-start gap-3">
        <span class="label w-16 shrink-0 pt-3">{{ severity }}</span>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="t in targets.filter((x) => x.set === shownSet && x.severity === severity)"
            :key="t.id"
            type="button"
            class="size-11 overflow-hidden rounded-[6px] border transition"
            :class="t.id === id ? 'border-queen-400 ring-1 ring-queen-400' : 'border-line hover:border-line-strong'"
            :title="`Image ${t.id} (${severityNote(t)})`"
            @click="pick(t.id)"
          >
            <img :src="t.png" :alt="`held-out image ${t.id}`" class="size-full" style="image-rendering: pixelated" />
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
