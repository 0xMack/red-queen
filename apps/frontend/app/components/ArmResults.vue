<script setup lang="ts">
// A tracked experiment's result, as the Learn chapters cite them (jobs/rl_experiment.py, jobs/snake_experiment.py):
// one row per arm, one dot per training run (a different rng seed), a bar at the arm's mean, dashed reference lines
// (a baseline heuristic, another method's score), and a table with each arm's comparison against the arm it differs
// from in one thing. The numbers are the caller's, hard-coded from a report: a chapter's figures shouldn't change
// under a reader because someone re-ran a job.
export interface ArmResult {
  key: string
  label: string
  sub?: string
  scores: number[]
  // the report's paired comparison against its baseline arm (same seeds paired); absent for the first arm
  versus?: { label: string; difference: number; p: number } | null
  highlight?: boolean
}

const props = withDefaults(
  defineProps<{
    eyebrow: string
    arms: ArmResult[]
    references?: { value: number; label: string; color: string }[]
    axis: [number, number]
    ticks: number[]
  }>(),
  { references: () => [] },
)

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const sd = (xs: number[]) => (xs.length > 1 ? Math.sqrt(xs.reduce((s, x) => s + (x - mean(xs)) ** 2, 0) / (xs.length - 1)) : 0)

const W = 460
const rowH = 26
const left = 150
const right = W - 12
const top = 27
const H = computed(() => top + props.arms.length * rowH + 20)
const x = (v: number) => left + ((v - props.axis[0]) / (props.axis[1] - props.axis[0])) * (right - left)
const color = (arm: ArmResult) => (arm.highlight ? "#ff5c7a" : "#60a5fa")
const signed = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`
</script>

<template>
  <figure class="card my-8 p-5" data-arm-results>
    <p class="eyebrow">{{ eyebrow }}</p>

    <svg :viewBox="`0 0 ${W} ${H}`" class="mt-3 block h-auto w-full" role="img" :aria-label="`Strip plot: ${eyebrow}`">
      <g v-for="t in ticks" :key="t">
        <line :x1="x(t)" :x2="x(t)" :y1="top - 6" :y2="top + arms.length * rowH" stroke="#222837" />
        <text :x="x(t)" :y="H - 4" text-anchor="middle" class="fill-fg-subtle font-mono text-[9px]">{{ t }}</text>
      </g>
      <g v-for="(r, i) in references" :key="r.label">
        <line :x1="x(r.value)" :x2="x(r.value)" :y1="top - 6" :y2="top + arms.length * rowH" :stroke="r.color" stroke-dasharray="4 3" stroke-opacity="0.8" />
        <text :x="x(r.value) + 4" :y="top - 7 - (i % 2) * 9" class="font-mono text-[9px]" :fill="r.color">{{ r.label }}</text>
      </g>
      <g v-for="(arm, i) in arms" :key="arm.key">
        <text :x="left - 8" :y="top + i * rowH + 16" text-anchor="end" class="font-mono text-[9px]" :class="arm.highlight ? 'fill-fg' : 'fill-fg-muted'">{{ arm.label }}</text>
        <line :x1="x(mean(arm.scores))" :x2="x(mean(arm.scores))" :y1="top + i * rowH + 5" :y2="top + i * rowH + 21" :stroke="color(arm)" stroke-width="2.5" />
        <circle
          v-for="(s, k) in arm.scores"
          :key="k"
          :cx="x(s)"
          :cy="top + i * rowH + 13"
          :r="arm.scores.length > 8 ? 2.5 : 3.5"
          :fill="color(arm)"
          fill-opacity="0.45"
          :stroke="color(arm)"
        />
      </g>
    </svg>

    <div class="mt-4 overflow-x-auto">
      <table class="w-full min-w-[440px] text-xs">
        <thead class="text-left text-[10px] tracking-wide text-fg-subtle uppercase">
          <tr>
            <th class="py-1 font-medium">variant</th>
            <th class="font-medium">runs</th>
            <th class="font-medium">score (mean ± sd)</th>
            <th class="font-medium">compared with</th>
            <th class="font-medium">p</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="arm in arms" :key="arm.key" class="border-t border-line">
            <td class="py-1.5 text-fg">{{ arm.label }}<span v-if="arm.sub" class="text-fg-subtle"> · {{ arm.sub }}</span></td>
            <td class="num text-fg-muted">{{ arm.scores.length }}</td>
            <td class="num text-fg">{{ mean(arm.scores).toFixed(2) }} ± {{ sd(arm.scores).toFixed(2) }}</td>
            <td class="num text-fg-muted">{{ arm.versus ? `${signed(arm.versus.difference)} vs ${arm.versus.label}` : "--" }}</td>
            <td class="num text-fg-muted">{{ arm.versus ? arm.versus.p.toFixed(2) : "--" }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <figcaption class="mt-3 text-xs text-fg-subtle"><slot /></figcaption>
  </figure>
</template>
