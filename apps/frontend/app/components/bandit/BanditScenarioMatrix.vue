<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"

// Every strategy on every scenario (docs/design/0011): skill, 0 (no better than random) to 100 (the best machine every
// pull), on held-out games -- one table where each scenario's trap shows as a dark cell in the row it catches. For
// Two lamps, both blind and seeing the lamp. Rows come from leaderboard records (the game page) or are passed in
// directly (`rows`, the Learn chapter's cited results); clicking a row selects that entrant.
export interface MatrixRow {
  id: string
  label: string
  /** Skill per column key. */
  skill: Record<string, number>
}

const props = withDefaults(defineProps<{ entries?: EvaluationRecord[]; rows?: MatrixRow[]; selectedId?: string | null; caption?: boolean }>(), {
  entries: () => [],
  rows: undefined,
  selectedId: null,
  caption: true,
})
const emit = defineEmits<{ select: [entrantId: string] }>()

// One column per scenario -- and a contextual one twice: blind, and seeing the lamp. A sequential game is only played
// seeing the room.
const COLUMNS = BANDIT_SCENARIOS.flatMap((s) =>
  s.sequential
    ? [{ key: s.id, title: s.title, sub: "sees the room" }]
    : s.contexts > 1
      ? [
          { key: s.id, title: s.title, sub: "blind" },
          { key: `${s.id}:lamp.v1`, title: s.title, sub: "sees the lamp" },
        ]
      : [{ key: s.id, title: s.title, sub: `${s.arms} × ${s.budget}` }],
)

const table = computed<MatrixRow[]>(
  () =>
    props.rows ??
    props.entries
      .filter((r) => r.metrics.bandit)
      .map((r) => ({ id: r.entrant_id, label: r.label, skill: Object.fromEntries(Object.entries(r.metrics.bandit!.scenarios).map(([k, v]) => [k, v.skill])) })),
)
// The best in each column, to mark.
const best = computed(() => Object.fromEntries(COLUMNS.map((c) => [c.key, Math.max(...table.value.map((r) => r.skill[c.key] ?? -Infinity))])))

function shade(v: number | undefined) {
  if (v === undefined) return "transparent"
  const t = Math.max(0, Math.min(1, v / 80))
  return v < 5 ? alpha("queen500", 0.18) : alpha("life400", 0.06 + 0.5 * t)
}
</script>

<template>
  <UiFigure kind="Every scenario" title="Which strategy wins depends on the game">
    <div class="overflow-x-auto">
      <table class="w-full border-separate border-spacing-0.5 text-xs">
        <thead>
          <tr>
            <th class="label pb-2 text-left font-normal">strategy</th>
            <th v-for="c in COLUMNS" :key="c.key" class="px-1 pb-2 text-center font-normal">
              <span class="block text-fg-muted">{{ c.title }}</span>
              <span class="block font-mono text-[10px] text-fg-subtle">{{ c.sub }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="r in table"
            :key="r.id"
            class="transition"
            :class="[entries.length ? 'cursor-pointer hover:brightness-125' : '', selectedId === r.id ? 'outline outline-1 outline-queen-400' : '']"
            @click="entries.length && emit('select', r.id)"
          >
            <th class="py-1 pr-3 text-left font-normal whitespace-nowrap" :class="selectedId === r.id ? 'text-fg' : 'text-fg-muted'">{{ r.label }}</th>
            <td
              v-for="c in COLUMNS"
              :key="c.key"
              class="num rounded-[4px] px-1 py-1.5 text-center"
              :class="r.skill[c.key] === best[c.key] ? 'font-semibold text-gold-300' : 'text-fg'"
              :style="{ background: shade(r.skill[c.key]) }"
            >
              {{ r.skill[c.key] === undefined ? "" : Math.round(r.skill[c.key]!) || 0 }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <template v-if="caption" #caption>
      Skill on 500 held-out games per scenario: 0 is no better than pulling at random, 100 is the best machine on every pull.
      Gold marks each column's best; red, strategies no better than random there.
    </template>
  </UiFigure>
</template>
