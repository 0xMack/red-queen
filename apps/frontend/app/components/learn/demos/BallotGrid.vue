<script setup lang="ts">
import { BALLOT_DEPTH, BALLOT_PLIES, openings } from "~/data/checkersBallot"

// The ballot (docs/design/0013): every Checkers position three plies from the start, one tile each, and the verdict of
// an 8-ply material search on it. The level ones (most) are the openings every strength measurement plays; the rest
// lose material by force for one side and are left out. Pick a tile to see its board and moves. The data is generated
// by jobs/export_ballot.py from the same code the measurement uses.

// The 32 playable squares in the board string's order: row by row from Red's side.
const SQUARES = Array.from({ length: 64 }, (_, i) => [i % 8, Math.floor(i / 8)] as const).filter(([x, y]) => (x + y) % 2 === 1)
const LABEL: Record<string, string> = { r: "red_man", R: "red_king", b: "black_man", B: "black_king" }

const cellsOf = (board: string) =>
  [...board].flatMap((c, i) => (c === "." ? [] : [{ x: SQUARES[i]![0], y: SQUARES[i]![1], label: LABEL[c]! }]))

const inBallot = openings.filter((o) => o.ballot).length
const firstRejected = openings.findIndex((o) => !o.ballot)
const selected = ref(firstRejected)
const current = computed(() => openings[selected.value]!)
const state = computed(() => ({ width: 8, height: 8, cells: cellsOf(current.value.board) }))
// Three plies: Red, Black, Red -- so it's Black to move, and the value is Black's.
const verdict = computed(() => {
  const v = current.value.value
  if (v === 0) return { text: "level: in the ballot", tone: "text-life-300" }
  return { text: `${v > 0 ? "Black" : "Red"} wins ${Math.abs(v) === 1 ? "a man" : `${Math.abs(v)} points of material`} by force: left out`, tone: "text-queen-300" }
})
</script>

<template>
  <UiFigure :title="`The ballot · ${inBallot} of ${openings.length} openings`">
    <div class="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)]">
      <div>
        <div class="grid grid-cols-[repeat(auto-fill,minmax(14px,1fr))] gap-[3px]" role="listbox" aria-label="Openings">
          <button
            v-for="(o, i) in openings"
            :key="i"
            type="button"
            role="option"
            :aria-selected="i === selected"
            :title="`${o.moves.join(' ')} · ${o.ballot ? 'level' : 'loses material'}`"
            class="aspect-square rounded-[3px] transition"
            :class="[
              o.ballot ? 'bg-life-400/45 hover:bg-life-400/80' : 'bg-queen-400/70 hover:bg-queen-400',
              i === selected ? 'ring-2 ring-fg ring-offset-1 ring-offset-surface' : '',
            ]"
            @click="selected = i"
          />
        </div>
        <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-fg-muted">
          <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-sm bg-life-400/60" />level after {{ BALLOT_DEPTH }} plies of material search ({{ inBallot }})</span>
          <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-sm bg-queen-400/70" />one side loses material by force ({{ openings.length - inBallot }})</span>
        </div>
      </div>
      <div class="min-w-0">
        <div class="mx-auto max-w-[280px]">
          <CheckersBoard :state="state" />
        </div>
        <p class="num mt-3 text-center text-sm text-fg">{{ current.moves.join("  ") }}</p>
        <p class="mt-1 text-center text-xs" :class="verdict.tone">{{ verdict.text }}</p>
      </div>
    </div>
    <template #caption>
      Every position {{ BALLOT_PLIES }} plies from the start (Red, Black, Red), one tile each, with two move orders reaching the same position
      counted once. Material search {{ BALLOT_DEPTH }} plies deep plays each out: the green ones stay level and make up the ballot; in the red
      ones a man is lost by force, and a game from there says more about the opening than about the players.
    </template>
  </UiFigure>
</template>
