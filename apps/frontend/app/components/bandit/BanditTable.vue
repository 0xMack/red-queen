<script setup lang="ts">
import type { Beliefs } from "~/utils/bandit"

// What a strategy believes, as the table it literally keeps (docs/design/0011): a row per situation it can tell
// apart -- one for a plain bandit, one per lamp colour when it sees the lamp -- and a column per machine. The same
// shape as a Q-table, which is the point: Snake's Q-table is this with 2,048 rows. Each cell: its estimate, how many
// pulls it's built on, and (per strategy) its uncertainty or the chance it's pulled next. The row in use is marked.
const props = withDefaults(
  defineProps<{
    beliefs: (Beliefs | null)[]
    /** The row the strategy is in now. */
    current?: number
    /** The machine it just pulled (or is about to). */
    arm?: number | null
    /** Row names: `["every pull"]`, or the lamp colours. */
    rowLabels?: string[]
    binary?: boolean
    /** What `values` are: estimates, or a gradient bandit's preferences. */
    kind?: "estimate" | "preference"
  }>(),
  { current: 0, arm: null, rowLabels: () => ["every pull"], binary: true, kind: "estimate" },
)

const arms = computed(() => props.beliefs.find((b) => b)?.values.length ?? 0)
// Colour by value within the table, so the best-looking cell in a row stands out whatever the payout scale.
const range = computed(() => {
  const all = props.beliefs.flatMap((b) => (b ? b.values.filter((_, a) => b.counts[a]! > 0 || props.kind === "preference") : []))
  return all.length ? { lo: Math.min(...all), hi: Math.max(...all) } : { lo: 0, hi: 1 }
})
function shade(b: Beliefs, a: number): string {
  if (props.kind === "estimate" && b.counts[a] === 0) return "transparent"
  const t = (b.values[a]! - range.value.lo) / Math.max(1e-9, range.value.hi - range.value.lo)
  return alpha("life400", 0.06 + 0.4 * t)
}
const fmt = (v: number) => (props.kind === "preference" ? v.toFixed(2) : props.binary ? v.toFixed(2) : v.toFixed(1))
</script>

<template>
  <div class="overflow-x-auto">
    <table class="w-full border-separate border-spacing-1 text-xs">
      <thead>
        <tr>
          <th class="label w-24 text-left font-normal">{{ beliefs.length > 1 ? "situation" : "" }}</th>
          <th v-for="a in arms" :key="a" class="font-display text-base font-normal" :style="{ color: armColor(a - 1) }">{{ armName(a - 1) }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(b, r) in beliefs" :key="r">
          <th class="text-left font-normal whitespace-nowrap" :class="r === current ? 'text-fg' : 'text-fg-subtle'">
            <span v-if="r === current" class="mr-1 text-queen-400">▸</span>{{ rowLabels[r] ?? `row ${r}` }}
          </th>
          <td
            v-for="a in arms"
            :key="a"
            class="num rounded-[5px] border px-1.5 py-1 text-center transition-colors duration-150"
            :class="r === current && arm === a - 1 ? 'border-fg/70' : r === current ? 'border-line-strong' : 'border-line'"
            :style="{ background: b ? shade(b, a - 1) : 'transparent' }"
          >
            <template v-if="b">
              <span class="block text-[13px]" :class="b.counts[a - 1] || kind === 'preference' ? 'text-fg' : 'text-fg-subtle'">
                {{ b.counts[a - 1] || kind === "preference" ? fmt(b.values[a - 1]!) : "?" }}
              </span>
              <span class="block text-[10px] text-fg-subtle">
                {{ b.counts[a - 1] }}×<template v-if="b.spread[a - 1]"> ±{{ b.spread[a - 1]!.toFixed(2) }}</template>
                <template v-if="b.probabilities.length"> · {{ Math.round(b.probabilities[a - 1]! * 100) }}%</template>
              </span>
            </template>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
