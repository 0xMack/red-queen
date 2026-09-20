<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"

// Head-to-head, for any versus game (docs/design/0007): every entrant against every other, from the
// row entrant's side -- the share of points it took (win 1, draw ½), shaded from lost-everything to
// won-everything. A single score hides *who* an entrant beats: this shows that a champion which beats
// random and 1-ply material still loses to 2-ply search. Reads `metrics.versus.by_opponent`, which
// jobs/evaluate_versus.py writes; entries without it are skipped.
const props = defineProps<{ entries: EvaluationRecord[]; selectedId?: string | null }>()
defineEmits<{ select: [entrantId: string] }>()

const rows = computed(() => props.entries.filter((r) => r.metrics.versus?.by_opponent))
const ids = computed(() => rows.value.map((r) => r.entrant_id))

interface Cell {
  share: number
  w: number
  d: number
  l: number
}
const cell = (row: EvaluationRecord, opponent: string): Cell | null => {
  const r = row.metrics.versus?.by_opponent[opponent]
  if (!r) return null
  const n = r.wins + r.draws + r.losses
  return { share: n ? (r.wins + 0.5 * r.draws) / n : 0.5, w: r.wins, d: r.draws, l: r.losses }
}
// 0 -> red (lost every point), 0.5 -> neutral, 1 -> green (took every point)
const shade = (share: number) =>
  share >= 0.5 ? `rgb(74 222 128 / ${((share - 0.5) * 2 * 0.75).toFixed(2)})` : `rgb(239 59 93 / ${((0.5 - share) * 2 * 0.75).toFixed(2)})`
const initials = (r: EvaluationRecord) => entrantShortLabel(r).replace(/[^A-Za-z0-9→ ]/g, "").split(/\s+/).slice(0, 3).join(" ")
</script>

<template>
  <section class="card overflow-x-auto p-5">
    <div class="flex flex-wrap items-baseline justify-between gap-3">
      <div>
        <h3 class="text-lg font-semibold">Head to head</h3>
        <p class="mt-1 max-w-3xl text-xs text-fg-subtle">
          Each row's share of the points against each column (win 1, draw ½). Green: the row took most of them;
          red: the column did. Hover a cell for wins / draws / losses.
        </p>
      </div>
      <div class="flex items-center gap-2 text-[11px] text-fg-subtle">
        loses all
        <span class="h-2.5 w-24 rounded-full" style="background: linear-gradient(90deg, rgb(239 59 93 / 0.75), transparent 50%, rgb(74 222 128 / 0.75))" />
        wins all
      </div>
    </div>
    <table class="mt-4 w-full border-separate border-spacing-1 text-xs">
      <thead>
        <tr>
          <th />
          <th v-for="c in rows" :key="c.entrant_id" class="max-w-24 px-1 pb-1 text-center align-bottom text-[10px] font-normal text-fg-subtle">
            <span class="line-clamp-2" :title="entrantShortLabel(c)">{{ initials(c) }}</span>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in rows" :key="r.entrant_id">
          <th
            class="cursor-pointer whitespace-nowrap py-1 pr-3 text-right text-[11px] font-normal"
            :class="selectedId === r.entrant_id ? 'text-queen-200' : 'text-fg-muted hover:text-fg'"
            @click="$emit('select', r.entrant_id)"
          >
            <span class="mr-1.5 inline-block size-2 rounded-full align-middle" :style="{ background: entrantColor(r) }" />{{ entrantShortLabel(r) }}
          </th>
          <td
            v-for="id in ids"
            :key="id"
            class="num h-8 min-w-12 rounded-md text-center"
            :class="id === r.entrant_id ? 'bg-sunken' : ''"
            :style="id !== r.entrant_id && cell(r, id) ? { background: shade(cell(r, id)!.share) } : undefined"
            :title="cell(r, id) ? `${entrantShortLabel(r)} vs ${entrantShortLabel(rows.find((x) => x.entrant_id === id)!)}: ${cell(r, id)!.w}W ${cell(r, id)!.d}D ${cell(r, id)!.l}L` : undefined"
          >
            <template v-if="id !== r.entrant_id && cell(r, id)">{{ Math.round(cell(r, id)!.share * 100) }}</template>
          </td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
