<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { CopyCatTarget, CopyCatTargetDetail } from "~/utils/copycat"

// Copy Cat's stage (docs/design/0022): pick a held-out image, or add your own, and see what the entrant on stage made of
// it -- its timelapse, ending on the SVG the leaderboard scored (CopyCatFilm) -- beside the corrupted input it was given
// and the clean SVG it was scored against. Your own image has no clean SVG and no score: it is traced on this device,
// by whichever entrant runs in the browser, and nothing is uploaded.
// Props spelled out (StageProps, games/types.ts): the SFC compiler can't resolve an imported type as the whole props.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()

const api = useApi()
const route = useRoute()
const router = useRouter()

const targets = ref<CopyCatTarget[]>([])
const failed = ref<string | null>(null)
const details = reactive(new Map<string, CopyCatTargetDetail>())

/** Your image, as a target with no clean SVG and no outputs; it stays in the picker once added, whoever is on stage. */
const { own, name: ownName } = useCopyCatOwnImage()

// an image is `set/seed` (seeds repeat across sets); a bare seed is a link from before there was more than one set
const id = computed<string | null>(() => {
  const raw = String(route.query.image ?? "")
  if (raw === OWN_IMAGE && own.value) return OWN_IMAGE
  const asked = raw.includes("/") ? raw : `synthetic/${raw}`
  if (targets.value.some((t) => t.id === asked)) return asked
  return targets.value.find((t) => t.severity === "medium")?.id ?? targets.value[0]?.id ?? null
})
const pick = (next: string) => router.replace({ query: { ...route.query, image: next } })

async function upload(file: File) {
  if (!file.type.startsWith("image/")) {
    failed.value = `${file.name} isn't an image`
    return
  }
  failed.value = null
  try {
    const pixels = await imageFilePixels(file)
    const canvas = document.createElement("canvas")
    canvas.width = pixels.width
    canvas.height = pixels.height
    canvas.getContext("2d")!.putImageData(pixels, 0, 0)
    ownName.value = file.name.replace(/\.[^.]+$/, "") || "yours"
    own.value = {
      id: OWN_IMAGE,
      set: OWN_IMAGE,
      seed: 0,
      source: null,
      severity: "clean",
      levels: {},
      width: pixels.width,
      height: pixels.height,
      png: canvas.toDataURL("image/png"),
      clean: "",
      protocol: "",
      outputs: {},
    }
    pick(OWN_IMAGE)
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
}

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
    if (s === null || s === OWN_IMAGE || details.has(s)) return
    try {
      details.set(s, await api.fetch<CopyCatTargetDetail>(`/games/copycat/targets/${s}`))
    } catch (e) {
      failed.value = e instanceof Error ? e.message : String(e)
    }
  },
  { immediate: true },
)

const detail = computed(() => (id.value === OWN_IMAGE ? own.value : id.value === null ? null : (details.get(id.value) ?? null)))
const output = computed(() => (props.entry && detail.value ? (detail.value.outputs[props.entry.entrant_id] ?? null) : null))

const severityNote = (t: CopyCatTarget) => {
  if (t.id === OWN_IMAGE) return `${t.width}×${t.height}, yours`
  const background = (t.levels.background as { kind?: string } | undefined)?.kind
  return [t.severity, background && t.severity !== "clean" ? `${background} background` : null].filter(Boolean).join(" · ")
}
</script>

<template>
  <div class="space-y-5">
    <p v-if="failed" class="text-sm text-fg-subtle">Couldn't load that image ({{ failed }}).</p>

    <CopyCatFilm
      :entrant-id="entry?.entrant_id ?? null"
      :entrant-label="entry?.label ?? 'This entrant'"
      :detail="detail"
      :result="output"
      :given-note="detail ? severityNote(detail) : undefined"
      :download-name="detail?.id === OWN_IMAGE ? ownName : undefined"
    >
      <template #picker>
        <CopyCatPicker :targets="targets" :selected="id" :own="own ? { png: own.png, name: ownName } : null" @pick="pick" @upload="upload" />
      </template>
    </CopyCatFilm>
  </div>
</template>
