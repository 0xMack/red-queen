<script setup lang="ts">
import type { components } from "~/types/api.gen"

// Copy Cat's research log (docs/design/0023): every `copycat.report`, what the kept ones add up to, and what one change
// did, image by image. Unlinked and local: the reports aren't part of the public site's static copy.
type Summary = components["schemas"]["ReportSummary"]

useHead({ title: "Copy Cat research log" })

const api = useApi()
const route = useRoute()
const router = useRouter()

const reports = ref<Summary[]>([])
const loaded = ref(false)
const failed = ref<string | null>(null)
const filter = ref<"all" | "kept" | "not_kept" | "unmarked">("all")

async function load() {
  try {
    reports.value = await api.fetch<Summary[]>("/games/copycat/reports")
  } catch (e) {
    failed.value = e instanceof Error ? e.message : String(e)
  }
  loaded.value = true
}
onMounted(load)

const kept = computed(() => reports.value.filter((r) => r.decision === "kept"))
const selected = computed(() => {
  const asked = route.query.report
  return typeof asked === "string" && reports.value.some((r) => r.name === asked) ? asked : (reports.value.at(-1)?.name ?? null)
})
const pick = (name: string) => router.replace({ query: { ...route.query, report: name } })

const GROUPS = ["synthetic/train", "icons/train", "icons/train-control", "synthetic/validate"]
const FILTERS = [
  { value: "all", label: "All" },
  { value: "kept", label: "Kept" },
  { value: "not_kept", label: "Not kept" },
  { value: "unmarked", label: "Unmarked" },
] as const
const rows = computed(() => {
  const named = new Map(reports.value.map((r) => [r.name, r]))
  return reports.value
    .filter((r) => filter.value === "all" || (r.decision ?? "unmarked") === filter.value)
    .map((r) => {
      const before = r.baseline ? named.get(r.baseline) : undefined
      return {
        report: r,
        cells: GROUPS.map((g) => {
          const now = r.groups[g]?.mean
          const was = before?.groups[g]?.mean
          return { group: g, now, change: now !== undefined && was !== undefined ? now - was : null }
        }),
      }
    })
    .reverse()
})
const signed = (change: number) => `${change >= 0 ? "+" : ""}${change.toFixed(3)}`
const changeTone = (change: number | null) =>
  change === null || Math.abs(change) < 0.0005 ? "text-fg-subtle" : change > 0 ? "text-life-300" : "text-queen-300"
</script>

<template>
  <main class="mx-auto max-w-[1600px] space-y-6 px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <UiSectionHeader :level="1" eyebrow="Research" title="Research log" :back="{ to: '/games/copycat', label: 'Copy Cat' }">
      Every change measured by <code class="chip">copycat.report</code>, whether it stayed, and what it did image by image.
    </UiSectionHeader>

    <UiEmpty v-if="!api.live">The research log needs a local backend: reports aren't part of the public site.</UiEmpty>
    <UiEmpty v-else-if="failed">The reports couldn't be loaded ({{ failed }}).</UiEmpty>
    <UiEmpty v-else-if="loaded && !reports.length">No reports yet. Run <code class="chip">copycat.report</code> with <code class="chip">params.out</code> set.</UiEmpty>
    <template v-else-if="loaded">
      <ReportProgress :kept="kept" :selected="selected" @select="pick" />
      <ReportChange v-if="selected" :name="selected" @decided="load" />

      <UiPanel label="Reports" title="Newest first" pad="none">
        <template #actions>
          <UiSegmented v-model="filter" :options="FILTERS" aria-label="Which reports" />
        </template>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="label text-left">
                <th class="px-4 py-2 font-normal">Report</th>
                <th class="px-2 py-2 font-normal">Decision</th>
                <th v-for="g in GROUPS" :key="g" class="px-2 py-2 text-right font-normal">{{ g }}</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="{ report, cells } in rows"
                :key="report.name"
                class="cursor-pointer border-t border-line transition hover:bg-sunken"
                :class="report.name === selected ? 'bg-sunken' : ''"
                @click="pick(report.name)"
              >
                <td class="max-w-md px-4 py-2">
                  <button type="button" class="font-mono text-xs text-fg" @click.stop="pick(report.name)">{{ report.name }}</button>
                  <UiBadge v-if="report.held_out" tone="gold" class="ml-1.5">held-out</UiBadge>
                  <p class="truncate text-xs text-fg-subtle" :title="report.note ?? undefined">{{ report.note }}</p>
                </td>
                <td class="px-2 py-2">
                  <UiBadge v-if="report.decision === 'kept'" tone="life" dot>kept</UiBadge>
                  <UiBadge v-else-if="report.decision === 'not_kept'">not kept</UiBadge>
                </td>
                <td v-for="cell in cells" :key="cell.group" class="num px-2 py-2 text-right">
                  <template v-if="cell.now !== undefined">
                    {{ cell.now.toFixed(2) }}
                    <span class="block text-[11px]" :class="changeTone(cell.change)">{{ cell.change === null ? "" : signed(cell.change) }}</span>
                  </template>
                  <span v-else class="text-fg-subtle">--</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </UiPanel>
    </template>
    <UiEmpty v-else compact class="h-40">Loading…</UiEmpty>
  </main>
</template>
