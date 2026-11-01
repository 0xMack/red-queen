<script setup lang="ts">
import type { components } from "~/types/api.gen"

// One change: what it did to each group against its baseline, then the images it moved, worst first. An image opens as
// it was given, as the baseline and this report traced it, and clean. Held-out images aren't opened one by one: they are
// scored once per change and never debugged on, so only their means are here.
type Detail = components["schemas"]["ReportDetail"]
type Image = components["schemas"]["ReportImage"]
type Decision = "kept" | "not_kept" | null

const props = defineProps<{ name: string }>()
const emit = defineEmits<{ decided: [] }>()

const api = useApi()
const report = ref<Detail | null>(null)
const failed = ref<string | null>(null)
const group = ref<string | null>(null)
const seed = ref<string | null>(null)
const image = ref<Image | null>(null)

const isHeldOut = (g: string) => g.endsWith("/heldout")

watch(
  () => props.name,
  async (name) => {
    report.value = null
    failed.value = null
    seed.value = null
    try {
      const loaded = await api.fetch<Detail>(`/games/copycat/reports/${name}`)
      if (name !== props.name) return
      report.value = loaded
      const groups = Object.keys(loaded.groups).filter((g) => !isHeldOut(g))
      group.value = groups.find((g) => g === group.value) ?? groups.find((g) => loaded.compared[g]?.down) ?? groups[0] ?? null
    } catch (e) {
      failed.value = e instanceof Error ? e.message : String(e)
    }
  },
  { immediate: true },
)

watch([seed, group], async ([s, g]) => {
  image.value = null
  if (!s || !g) return
  const wanted = `/games/copycat/reports/${props.name}/images/${g}/${s}`
  try {
    const loaded = await api.fetch<Image>(wanted)
    if (s === seed.value && g === group.value) image.value = loaded
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
})

const rows = computed(() => {
  const r = report.value
  if (!r) return []
  return Object.entries(r.groups).map(([g, row]) => ({ group: g, mean: row.mean, worst: row.worst, lr: row.columns.layer_recovery, compared: r.compared[g] ?? null }))
})
const groupOptions = computed(() => Object.keys(report.value?.groups ?? {}).filter((g) => !isHeldOut(g)))
/** The images this group moved, worst first. */
const movers = computed(() => {
  const moved = report.value && group.value ? report.value.compared[group.value]?.moved : null
  return Object.entries(moved ?? {}).map(([s, change]) => ({ seed: s, value: change })).sort((a, b) => a.value - b.value)
})
/** The lowest-scoring images, whether this change moved them or not. */
const worstImages = computed(() => {
  const per = report.value && group.value ? report.value.per_image[group.value] : null
  return per ? Object.entries(per).map(([s, dB]) => ({ seed: s, dB })).sort((a, b) => a.dB - b.dB).slice(0, 12) : []
})

const DECISIONS = [
  { value: "kept", label: "Kept" },
  { value: "not_kept", label: "Not kept" },
  { value: null, label: "Unmarked" },
] as const
const decision = computed<Decision>({
  get: () => report.value?.decision ?? null,
  set: async (value) => {
    if (!report.value) return
    report.value.decision = value
    await api.fetch(`/games/copycat/reports/${props.name}`, { method: "PATCH", body: { decision: value } })
    emit("decided")
  },
})
</script>

<template>
  <UiPanel label="One change" :title="name">
    <template #actions>
      <UiSegmented v-model="decision" :options="DECISIONS" aria-label="Whether this change stays" />
    </template>
    <p v-if="failed" class="text-sm text-fg-subtle">This report couldn't be loaded ({{ failed }}).</p>
    <div v-else-if="report" class="space-y-5">
      <p class="text-sm text-fg-muted">
        {{ report.note ?? "No note." }}
        <span class="font-mono text-xs text-fg-subtle">{{ report.commit }} · {{ formatTimestamp(report.written_at) }}</span>
      </p>
      <div class="flex flex-wrap items-center gap-2 text-xs text-fg-subtle">
        <span v-if="report.baseline">Against <span class="font-mono">{{ report.baseline }}</span></span>
        <span v-else>No baseline</span>
        <UiBadge v-if="report.held_out" tone="gold" dot>looked at held-out</UiBadge>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="label text-left">
              <th class="py-1.5 pr-4 font-normal">Group</th>
              <th class="py-1.5 pr-4 text-right font-normal">Before</th>
              <th class="py-1.5 pr-4 text-right font-normal">After</th>
              <th class="py-1.5 pr-4 text-right font-normal">Up / down</th>
              <th class="py-1.5 pr-4 text-right font-normal">Worst</th>
              <th class="py-1.5 text-right font-normal">Layer recovery</th>
            </tr>
          </thead>
          <tbody class="num">
            <tr v-for="row in rows" :key="row.group" class="border-t border-line">
              <td class="py-1.5 pr-4 font-mono text-xs">
                {{ row.group }}
                <UiBadge v-if="isHeldOut(row.group)" tone="gold">held-out</UiBadge>
              </td>
              <td class="py-1.5 pr-4 text-right text-fg-subtle">{{ row.compared?.before.toFixed(3) ?? "--" }}</td>
              <td class="py-1.5 pr-4 text-right">{{ row.mean.toFixed(3) }}</td>
              <td class="py-1.5 pr-4 text-right">{{ row.compared ? `${row.compared.up} / ${row.compared.down}` : "--" }}</td>
              <td class="py-1.5 pr-4 text-right">{{ row.worst.toFixed(2) }}</td>
              <td class="py-1.5 text-right">{{ row.lr?.toFixed(3) ?? "--" }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="group" class="space-y-3">
        <UiSegmented v-model="group" :options="groupOptions.map((g) => ({ value: g, label: g }))" aria-label="Which group's images" />
        <div class="flex flex-wrap gap-1.5">
          <template v-if="report.baseline">
            <button
              v-for="m in movers"
              :key="m.seed"
              type="button"
              class="rounded-[6px] border px-2 py-1 font-mono text-[11px] transition"
              :class="[seed === m.seed ? 'border-queen-400' : 'border-line hover:border-line-strong', m.value < 0 ? 'text-queen-300' : 'text-life-300']"
              @click="seed = m.seed"
            >
              {{ m.seed }}: {{ formatSigned(m.value) }}
            </button>
            <span v-if="!movers.length" class="text-xs text-fg-subtle">No image in this group moved by more than 0.05 dB.</span>
          </template>
        </div>
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="label mr-1">Lowest</span>
          <button
            v-for="m in worstImages"
            :key="m.seed"
            type="button"
            class="rounded-[6px] border px-2 py-1 font-mono text-[11px] text-fg-muted transition"
            :class="seed === m.seed ? 'border-queen-400' : 'border-line hover:border-line-strong'"
            @click="seed = m.seed"
          >
            {{ m.seed }}: {{ m.dB.toFixed(1) }} dB
          </button>
        </div>
        <div v-if="seed" class="grid gap-4 sm:grid-cols-4">
          <CopyCatImage label="Given" :src="image?.png" :note="image?.severity" />
          <CopyCatImage label="Baseline" :svg="image?.baseline_svg" :note="image?.baseline_fidelity != null ? `${image.baseline_fidelity.toFixed(2)} dB` : undefined">
            <template #empty>{{ image ? "That report kept no SVGs" : "Loading…" }}</template>
          </CopyCatImage>
          <CopyCatImage label="This change" :svg="image?.svg" :note="image?.fidelity != null ? `${image.fidelity.toFixed(2)} dB` : undefined">
            <template #empty>{{ image ? "This report kept no SVGs" : "Loading…" }}</template>
          </CopyCatImage>
          <CopyCatImage label="Clean" :svg="image?.clean" />
        </div>
      </div>
    </div>
    <UiEmpty v-else compact class="h-40">Loading…</UiEmpty>
  </UiPanel>
</template>
