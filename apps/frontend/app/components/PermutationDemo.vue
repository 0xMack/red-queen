<script setup lang="ts">
// Why neuroevolution has no crossover, made concrete. Network B is network A with two hidden units swapped
// -- literally the same function (identical XOR outputs, identical fitness), but a different list of numbers,
// because "hidden unit 0" is just a name. Breed them by averaging or splicing the lists and the child is
// garbage: each gene now means something different in each parent. This is the *competing conventions*
// (permutation) problem, and it's the exact thing NEAT's innovation numbers exist to solve.
import { XOR_CASES } from "~/utils/neat"
import { output, XOR_SHAPE, xorWeightsFitness } from "~/utils/neuro"

const A = [3.96, -1.73, 4.86, 2.42, 1.87, 0.92, -0.8, -1.33, -2.09, -0.8, 3.96, -3.45, -2.24]
// swap hidden units 0 and 1: their input weights (0-1 <-> 2-3), biases (6 <-> 7), and output weights (9 <-> 10)
const SWAPPED = (() => {
  const b = [...A]
  ;[b[0], b[1], b[2], b[3]] = [A[2]!, A[3]!, A[0]!, A[1]!]
  ;[b[6], b[7]] = [A[7]!, A[6]!]
  ;[b[9], b[10]] = [A[10]!, A[9]!]
  return b
})()
// the "same order" partner: A nudged a little, hidden units still in the same slots
const NEARBY = A.map((w, i) => w + Math.sin(i * 2.3) * 0.35)

const partner = ref<"swapped" | "nearby">("swapped")
const method = ref<"average" | "splice">("average")

const B = computed(() => (partner.value === "swapped" ? SWAPPED : NEARBY))
const child = computed(() => (method.value === "average" ? A.map((w, i) => (w + B.value[i]!) / 2) : A.map((w, i) => (i < 6 ? w : B.value[i]!))))

const rows = computed(() => [
  { name: "parent A", genome: A, note: "" },
  { name: partner.value === "swapped" ? "parent B (A with hidden 0 ↔ 1 swapped)" : "parent B (A, nudged)", genome: B.value, note: "" },
  { name: `child (${method.value})`, genome: child.value, note: "" },
])

const fit = (g: number[]) => xorWeightsFitness(g)
const outs = (g: number[]) => XOR_CASES.map((c) => (output(g, XOR_SHAPE, c.input) + 1) / 2)

function cellColor(w: number): string {
  const t = Math.min(1, Math.abs(w) / 5)
  return w >= 0 ? `rgb(74 222 128 / ${0.1 + 0.5 * t})` : `rgb(255 92 122 / ${0.1 + 0.5 * t})`
}
</script>

<template>
  <figure class="card my-8 p-5">
    <p class="eyebrow">Competing conventions · try breeding two equally good networks</p>
    <div class="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
      <fieldset class="flex items-center gap-2">
        <legend class="sr-only">Partner</legend>
        <span class="text-fg-muted">B is</span>
        <button class="rounded-full border px-3 py-1 text-xs transition" :class="partner === 'swapped' ? 'border-queen-400/50 bg-queen-500/10 text-queen-200' : 'border-line text-fg-muted'" @click="partner = 'swapped'">A, hidden units swapped</button>
        <button class="rounded-full border px-3 py-1 text-xs transition" :class="partner === 'nearby' ? 'border-queen-400/50 bg-queen-500/10 text-queen-200' : 'border-line text-fg-muted'" @click="partner = 'nearby'">A, nudged</button>
      </fieldset>
      <fieldset class="flex items-center gap-2">
        <legend class="sr-only">Crossover</legend>
        <span class="text-fg-muted">breed by</span>
        <button class="rounded-full border px-3 py-1 text-xs transition" :class="method === 'average' ? 'border-queen-400/50 bg-queen-500/10 text-queen-200' : 'border-line text-fg-muted'" @click="method = 'average'">averaging</button>
        <button class="rounded-full border px-3 py-1 text-xs transition" :class="method === 'splice' ? 'border-queen-400/50 bg-queen-500/10 text-queen-200' : 'border-line text-fg-muted'" @click="method = 'splice'">splicing</button>
      </fieldset>
    </div>

    <div class="mt-4 space-y-3">
      <div v-for="r in rows" :key="r.name" class="rounded-lg border border-line bg-sunken p-3">
        <div class="flex flex-wrap items-baseline justify-between gap-2 text-xs">
          <span class="text-fg-muted">{{ r.name }}</span>
          <span>
            fitness <span class="num text-sm" :class="fit(r.genome) > 0.95 ? 'text-life-300' : 'text-queen-300'">{{ fit(r.genome).toFixed(3) }}</span>
            <span class="num ml-3 text-fg-subtle">outputs {{ outs(r.genome).map((v) => v.toFixed(2)).join("  ") }}</span>
          </span>
        </div>
        <div class="mt-2 flex flex-wrap gap-1">
          <span v-for="(w, i) in r.genome" :key="i" class="num inline-flex h-7 w-10 items-center justify-center rounded text-[10px] text-fg" :style="{ background: cellColor(w) }">{{ w.toFixed(1) }}</span>
        </div>
      </div>
    </div>
    <figcaption class="mt-3 text-xs text-fg-subtle">
      XOR should read <span class="num">0 1 1 0</span>. With the swapped partner, A and B are the <em>same function</em> (identical outputs)
      yet their child is broken; with the nudged partner, the child is fine, because gene <em>i</em> means the same thing in both. Same
      operator, opposite outcome: whether crossover helps depends entirely on whether the parents' genes line up.
    </figcaption>
  </figure>
</template>
