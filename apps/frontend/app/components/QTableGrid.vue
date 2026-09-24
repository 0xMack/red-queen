<script setup lang="ts">
// Snake's Q-table (features.v1: 11 binary features -> 2,048 rows), drawn as the 256 rows the game can actually reach:
// one 8 x 8 grid per heading, a row per danger pattern (ahead / left / right), a column per food direction. Each cell
// is a state, colored by the move the table currently prefers there, faded while it has never been updated; the
// state the demo snake is in right now is outlined. docs/design/0010 Phase 1b.
//
// Row index = sum of 2^i over the set features, in features.v1's order: danger ahead/left/right (bits 0-2), heading
// right/down/left/up (bits 3-6, one-hot), food left/right/up/down (bits 7-10).
const props = defineProps<{
  values: Float64Array | number[] | null
  visits?: Uint32Array | number[] | null
  current?: number | null
  // Without visit counts (a recorded table), a row still at its starting value counts as never updated
  initial?: number
}>()

const ACTIONS = ["turn left", "straight", "turn right"] as const
const ACTION_COLORS = ["#60a5fa", "#4ade80", "#fbbf24"]
const HEADINGS = [
  { bit: 3, label: "→", name: "heading right" },
  { bit: 4, label: "↓", name: "heading down" },
  { bit: 5, label: "←", name: "heading left" },
  { bit: 6, label: "↑", name: "heading up" },
]
// danger ahead / left / right, as a bit pattern 0..7
const DANGERS = Array.from({ length: 8 }, (_, d) => ({
  bits: d,
  label: ["A", "L", "R"].filter((_, i) => d & (1 << i)).join("") || "·",
}))
// food: left (7) / right (8) x up (9) / down (10), never "on the head"
const FOODS = [
  { mask: (1 << 7) | (1 << 9), label: "↖" },
  { mask: 1 << 9, label: "↑" },
  { mask: (1 << 8) | (1 << 9), label: "↗" },
  { mask: 1 << 7, label: "←" },
  { mask: 1 << 8, label: "→" },
  { mask: (1 << 7) | (1 << 10), label: "↙" },
  { mask: 1 << 10, label: "↓" },
  { mask: (1 << 8) | (1 << 10), label: "↘" },
]

function describe(row: number): string {
  const heading = HEADINGS.find((h) => row & (1 << h.bit))?.name ?? "?"
  const danger = ["ahead", "left", "right"].filter((_, i) => row & (1 << i))
  const food = FOODS.find((f) => (row & 0b11110000000) === f.mask)?.label ?? "?"
  return `${heading}, danger ${danger.length ? danger.join(" + ") : "none"}, food ${food}`
}

interface Cell {
  row: number
  action: number | null // the greedy action, null while never updated
  q: [number, number, number] | null
  visits: number
}

const blocks = computed(() =>
  HEADINGS.map((heading) => ({
    heading,
    rows: DANGERS.map((danger) =>
      FOODS.map((food): Cell => {
        const row = danger.bits | (1 << heading.bit) | food.mask
        const visits = props.visits ? Number(props.visits[row] ?? 0) : 0
        if (!props.values || props.values.length < (row + 1) * 3) return { row, action: null, q: null, visits }
        const q: [number, number, number] = [props.values[row * 3]!, props.values[row * 3 + 1]!, props.values[row * 3 + 2]!]
        let best = 0
        for (let a = 1; a < 3; a++) if (q[a]! > q[best]!) best = a
        const initial = props.initial ?? 0
        const learned = props.visits ? visits > 0 : q.some((v) => v !== initial)
        return { row, action: learned ? best : null, q, visits }
      }),
    ),
  })),
)

const counts = computed(() => {
  const all = blocks.value.flatMap((b) => b.rows.flat())
  return { learned: all.filter((c) => c.action !== null).length, total: all.length }
})

function title(cell: Cell): string {
  const q = cell.q ? ` -- Q: ${cell.q.map((v, i) => `${ACTIONS[i]} ${v.toFixed(2)}`).join(", ")}` : ""
  const learned = cell.action === null ? " (never updated)" : ` -> ${ACTIONS[cell.action]}`
  return `row ${cell.row}: ${describe(cell.row)}${learned}${q}${props.visits ? ` · ${cell.visits} updates` : ""}`
}
</script>

<template>
  <div data-q-table-grid>
    <div class="grid grid-cols-2 gap-4 sm:grid-cols-4">
      <div v-for="block in blocks" :key="block.heading.bit">
        <p class="mb-1 text-center text-xs text-fg-subtle">
          <span class="text-base text-fg">{{ block.heading.label }}</span> {{ block.heading.name }}
        </p>
        <div class="grid grid-cols-[1.6rem_repeat(8,minmax(0,1fr))] gap-px text-[9px] leading-none">
          <span />
          <span v-for="f in FOODS" :key="f.mask" class="pb-0.5 text-center text-fg-subtle" :title="`food ${f.label}`">{{ f.label }}</span>
          <template v-for="(cells, d) in block.rows" :key="d">
            <span class="flex items-center justify-end pr-1 font-mono text-fg-subtle" :title="`danger: ${DANGERS[d]!.label}`">{{ DANGERS[d]!.label }}</span>
            <span
              v-for="cell in cells"
              :key="cell.row"
              class="aspect-square rounded-[2px] transition-colors"
              :class="cell.row === current ? 'ring-2 ring-fg ring-offset-1 ring-offset-surface' : ''"
              :style="{
                background: cell.action === null ? 'var(--color-raised, #1b2130)' : ACTION_COLORS[cell.action],
                opacity: cell.action === null ? 0.45 : 1,
              }"
              :title="title(cell)"
            />
          </template>
        </div>
      </div>
    </div>
    <div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-subtle">
      <span v-for="(a, i) in ACTIONS" :key="a" class="flex items-center gap-1.5">
        <span class="size-2.5 rounded-[2px]" :style="{ background: ACTION_COLORS[i] }" />{{ a }}
      </span>
      <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-[2px] bg-raised opacity-45" />not learned yet</span>
      <span class="ml-auto num">{{ counts.learned }} of {{ counts.total }} reachable states learned</span>
    </div>
  </div>
</template>
