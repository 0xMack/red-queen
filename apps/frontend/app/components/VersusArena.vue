<script setup lang="ts">
import type { Tally } from "~/types/versus"

// Measure two strategies against each other at full speed, off the visible board -- game-agnostic:
// the session plays the games on fresh engines and reports each pairing from A's point of view.
const props = defineProps<{
  strategies: { id: string; label: string }[]
  run: (a: string, b: string, games: number, onProgress: (played: number, tally: Tally) => void) => Promise<Tally>
  disabled?: boolean
  defaultA?: string
  defaultB?: string
  /** What the measurement is comparable to, shown under the heading. */
  note?: string
}>()

const a = ref(props.defaultA ?? props.strategies[0]?.id ?? "")
const b = ref(props.defaultB ?? props.strategies[1]?.id ?? "")
const games = ref(40)
const running = ref(false)
const played = ref(0)
const result = ref<{ a: string; b: string; tally: Tally; games: number } | null>(null)
const label = (id: string) => props.strategies.find((s) => s.id === id)?.label ?? id

async function go() {
  const [idA, idB, count] = [a.value, b.value, games.value]
  running.value = true
  played.value = 0
  result.value = null
  const done = await props.run(idA, idB, count, (n, tally) => {
    played.value = n
    result.value = { a: label(idA), b: label(idB), tally, games: count }
  })
  result.value = { a: label(idA), b: label(idB), tally: done, games: count }
  running.value = false
}
</script>

<template>
  <section class="card p-5">
    <p class="eyebrow">Arena</p>
    <h3 class="mt-1 font-display text-lg font-semibold">Measure two strategies against each other</h3>
    <p class="mt-1 max-w-3xl text-sm text-fg-muted">
      Plays the games right here at full speed, alternating who moves first so first-move advantage
      cancels out.
      <slot>{{ note }}</slot>
    </p>
    <div class="mt-4 flex flex-wrap items-end gap-3 text-sm">
      <label>
        <span class="eyebrow">A</span>
        <select v-model="a" class="mt-1 block rounded-lg border border-line bg-surface px-3 py-2" :disabled="running">
          <option v-for="s in strategies" :key="s.id" :value="s.id">{{ s.label }}</option>
        </select>
      </label>
      <span class="pb-2 text-fg-subtle">vs</span>
      <label>
        <span class="eyebrow">B</span>
        <select v-model="b" class="mt-1 block rounded-lg border border-line bg-surface px-3 py-2" :disabled="running">
          <option v-for="s in strategies" :key="s.id" :value="s.id">{{ s.label }}</option>
        </select>
      </label>
      <label>
        <span class="eyebrow">Games</span>
        <select v-model.number="games" class="mt-1 block rounded-lg border border-line bg-surface px-3 py-2" :disabled="running">
          <option v-for="n in [10, 40, 100, 200]" :key="n" :value="n">{{ n }}</option>
        </select>
      </label>
      <button class="rounded-lg bg-queen-500 px-4 py-2 font-medium text-white disabled:opacity-50" :disabled="running || disabled" @click="go">
        {{ running ? `Playing… ${played}/${games}` : "Run" }}
      </button>
    </div>
    <p v-if="result" class="mt-4 text-sm" data-testid="arena-result">
      <strong>{{ result.a }}</strong>
      <span class="num"> {{ result.tally.wins[0] }} wins</span>,
      <span class="num">{{ result.tally.draws }} draws</span>,
      <span class="num">{{ result.tally.wins[1] }} losses</span>
      against <strong>{{ result.b }}</strong>
      <span class="text-fg-subtle">({{ played }}/{{ result.games }} games)</span>
    </p>
  </section>
</template>
