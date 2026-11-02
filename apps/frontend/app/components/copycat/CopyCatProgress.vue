<script setup lang="ts">
import { COPYCAT_STAGES, reportGroups } from "~/data/copycatStages"
import type { ChartSeries } from "~/types/chart"

// Copy Cat's game page, `GameModule.Insights`: what the kept changes have added up to, each group's mean over them with
// the staged plan's targets, and the changes themselves. Each is a run (`config.report`, docs/design/0023) and opens as
// one. Only the groups every kept report measured are drawn, since the chart has no gaps (`synthetic/validate` joined
// the reports later). A change that didn't stay is listed only if it is marked instructive.
defineOptions({ inheritAttrs: false })

const runsStore = useRunsStore()
onMounted(() => runsStore.ensureLoaded())

const COLOURS = [palette.queen400, palette.signal400, palette.life400, palette.gold400, palette.queen200, palette.signal300]
const TABLE_GROUPS = ["synthetic/train", "icons/train", "icons/train-control", "synthetic/validate"]

const reports = computed(() =>
  runsStore.runs.filter((r) => typeof r.config?.report === "string" && r.status === "completed").sort((a, b) => a.created_at - b.created_at),
)
const kept = computed(() => reports.value.filter((r) => r.config?.decision === "kept"))
const unmarkedOrRejected = computed(() => reports.value.filter((r) => r.config?.decision !== "kept" && r.config?.decision !== "instructive").length)

const x = computed(() => kept.value.map((_, i) => i + 1))
const series = computed<ChartSeries[]>(() => {
  const first = kept.value[0]
  if (!first) return []
  const groups = Object.keys(reportGroups(first)).filter((g) => kept.value.every((r) => g in reportGroups(r)))
  const lines = groups.map((group, i) => ({
    key: group,
    label: group,
    color: COLOURS[i % COLOURS.length]!,
    values: kept.value.map((r) => reportGroups(r)[group]!.mean),
  }))
  const targets = COPYCAT_STAGES.filter((s) => s.group && groups.includes(s.group)).map((s) => ({
    key: `stage-${s.stage}`,
    label: `Stage ${s.stage} target`,
    color: lines[groups.indexOf(s.group!)]!.color,
    values: kept.value.map(() => s.target),
    dashed: true,
    width: 1,
  }))
  return [...lines, ...targets]
})
const stages = computed(() => {
  const latest = kept.value.at(-1)
  return COPYCAT_STAGES.map((s) => {
    const value = latest ? s.measure(reportGroups(latest)) : undefined
    return {
      label: `Stage ${s.stage}: ${s.goal}`,
      value: value === undefined ? "not measured" : value.toFixed(s.target < 1 ? 3 : 2),
      tone: value !== undefined && value >= s.target ? ("life" as const) : ("default" as const),
    }
  })
})

/** Newest first, each with what it changed against its baseline. */
const rows = computed(() => {
  const named = new Map(reports.value.map((r) => [r.config!.report as string, r]))
  return reports.value
    .filter((r) => r.config?.decision === "kept" || r.config?.decision === "instructive")
    .map((run) => {
      const before = typeof run.config?.baseline === "string" ? named.get(run.config.baseline) : undefined
      return {
        run,
        cells: TABLE_GROUPS.map((group) => {
          const now = reportGroups(run)[group]?.mean
          const was = before ? reportGroups(before)[group]?.mean : undefined
          return { group, now, change: now !== undefined && was !== undefined ? now - was : null }
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
  <UiPanel v-if="kept.length" label="Progress" title="Over the kept changes" ticks>
    <div class="space-y-4">
      <UiStats :items="stages" />
      <LineChart
        :x="x"
        :series="series"
        x-label="kept change"
        clickable
        :format="(v) => v.toFixed(2)"
        @select="(i) => navigateTo(`/runs/${kept[i - 1]!.run_id}`)"
      />
      <UiLegend :series="series" />
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="label text-left">
              <th class="py-2 pr-3 font-normal">Change</th>
              <th v-for="g in TABLE_GROUPS" :key="g" class="px-2 py-2 text-right font-normal">{{ g }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="{ run, cells } in rows" :key="run.run_id" class="border-t border-line">
              <td class="max-w-md py-2 pr-3">
                <NuxtLink :to="`/runs/${run.run_id}`" class="font-mono text-xs text-fg hover:text-queen-300">{{ run.config?.report }}</NuxtLink>
                <UiBadge v-if="run.config?.decision === 'instructive'" tone="gold" class="ml-1.5">tried, not kept</UiBadge>
                <p class="text-xs text-fg-subtle">{{ run.config?.note }}</p>
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
      <p class="text-xs text-fg-subtle">
        <NuxtLink to="/learn/earning-complexity" class="underline hover:text-fg">Earning Each Piece of Complexity</NuxtLink> tells the story behind these changes.
      </p>
      <p v-if="unmarkedOrRejected" class="text-xs text-fg-subtle">
        {{ unmarkedOrRejected }} more {{ unmarkedOrRejected === 1 ? "report is" : "reports are" }} under
        <NuxtLink to="/runs" class="underline hover:text-fg">Runs</NuxtLink>, unmarked or not kept.
      </p>
    </div>
  </UiPanel>
</template>
