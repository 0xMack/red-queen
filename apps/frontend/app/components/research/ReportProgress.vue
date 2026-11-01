<script setup lang="ts">
import { COPYCAT_STAGES } from "~/data/copycatStages"
import type { components } from "~/types/api.gen"
import type { ChartSeries } from "~/types/chart"

// What the kept changes have added up to: each group's mean over them, with the staged plan's targets. Only the groups
// every kept report measured are drawn, since the chart has no gaps (`synthetic/validate` joined the reports later).
type Summary = components["schemas"]["ReportSummary"]
const props = defineProps<{ kept: Summary[]; selected: string | null }>()
const emit = defineEmits<{ select: [name: string] }>()

const COLOURS = [palette.queen400, palette.signal400, palette.life400, palette.gold400, palette.queen200, palette.signal300]

const x = computed(() => props.kept.map((_, i) => i + 1))
const series = computed<ChartSeries[]>(() => {
  const first = props.kept[0]
  if (!first) return []
  const groups = Object.keys(first.groups).filter((g) => props.kept.every((r) => g in r.groups))
  const lines = groups.map((group, i) => ({
    key: group,
    label: group,
    color: COLOURS[i % COLOURS.length]!,
    values: props.kept.map((r) => r.groups[group]!.mean),
  }))
  const targets = COPYCAT_STAGES.filter((s) => s.group && groups.includes(s.group)).map((s) => ({
    key: `stage-${s.stage}`,
    label: `Stage ${s.stage} target`,
    color: lines[groups.indexOf(s.group!)]!.color,
    values: props.kept.map(() => s.target),
    dashed: true,
    width: 1,
  }))
  return [...lines, ...targets]
})
const marker = computed(() => {
  const i = props.kept.findIndex((r) => r.name === props.selected)
  return i < 0 ? null : i + 1
})
const latest = computed(() => props.kept.at(-1) ?? null)
const stages = computed(() =>
  COPYCAT_STAGES.map((s) => {
    const value = latest.value ? s.measure(latest.value) : undefined
    return {
      label: `Stage ${s.stage}: ${s.goal}`,
      value: value === undefined ? "not measured" : value.toFixed(s.target < 1 ? 3 : 2),
      tone: value !== undefined && value >= s.target ? ("life" as const) : ("default" as const),
    }
  }),
)
</script>

<template>
  <UiPanel label="Progress" title="Over the kept changes" ticks>
    <div v-if="kept.length" class="space-y-4">
      <UiStats :items="stages" />
      <LineChart
        :x="x"
        :series="series"
        :marker="marker"
        x-label="kept change"
        clickable
        :format="(v) => v.toFixed(2)"
        @select="(i) => emit('select', kept[i - 1]!.name)"
      />
      <UiLegend :series="series" />
    </div>
    <UiEmpty v-else compact>No report is marked kept yet.</UiEmpty>
  </UiPanel>
</template>
