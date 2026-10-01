<script setup lang="ts">
import * as F from "~/data/math/training-regimes"

// Prioritized fictitious self-play (docs/design/0014), by hand: a pool of past selves, the network's running score
// against each (drag them), and the share of pool games each gets -- uniformly, or weighted by (1 - s)^p + 0.01 as
// libs/rl/rust/envs/src/selfplay.rs draws them. The weight formula is worked for the hovered (or first) past self.
const ages = ["−50k", "−45k", "−40k", "−35k", "−30k", "−25k", "−20k", "−15k", "−10k", "−5k"]
const scores = ref([0.97, 0.95, 0.9, 0.93, 0.85, 0.8, 0.72, 0.6, 0.55, 0.5])
const p = ref(2)
const focus = ref(9)

const weights = computed(() => scores.value.map((s) => (p.value === 0 ? 1 : (1 - s) ** p.value + 0.01)))
const shares = computed(() => {
  const total = weights.value.reduce((a, b) => a + b, 0)
  return weights.value.map((w) => w / total)
})

const scope = useOrProvideTermScope()
watch(
  [scores, p, focus],
  () => {
    scope.values.value = { "s-i": scores.value[focus.value]!, p: p.value }
    scope.expected.value = p.value === 0 ? {} : { "w-i": weights.value[focus.value]! }
  },
  { immediate: true, deep: true },
)
const PRESETS = [
  { label: "Left them behind", scores: [0.97, 0.95, 0.9, 0.93, 0.85, 0.8, 0.72, 0.6, 0.55, 0.5] },
  { label: "Forgot how to beat an old one", scores: [0.97, 0.95, 0.3, 0.93, 0.85, 0.8, 0.72, 0.6, 0.55, 0.5] },
  { label: "All even", scores: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] },
]
</script>

<template>
  <UiFigure title="Prioritized opponents: who gets the games">
    <div class="flex flex-wrap items-center gap-2">
      <button v-for="pr in PRESETS" :key="pr.label" class="btn-ghost btn-sm" @click="scores = [...pr.scores]">{{ pr.label }}</button>
      <UiSegmented
        v-model="p"
        class="ml-auto"
        :options="[
          { value: 0, label: 'uniform' },
          { value: 1, label: 'p = 1' },
          { value: 2, label: 'p = 2' },
          { value: 4, label: 'p = 4' },
        ]"
        aria-label="Prioritization"
      />
    </div>
    <div class="math-panel mt-4">
      <MathFormula :formula="F.weight" bare />
      <MathFormula class="mt-1" :formula="F.choice" bare />
    </div>
    <div class="mt-4 grid grid-cols-10 gap-2">
      <div v-for="(s, i) in scores" :key="i" class="min-w-0 text-center" @mouseenter="focus = i">
        <div class="relative mx-auto flex h-32 w-full items-end justify-center overflow-hidden rounded-md bg-sunken" :class="i === focus ? 'ring-1 ring-queen-400' : ''">
          <div class="w-full rounded-t-sm bg-queen-400/80 transition-all" :style="{ height: `${(shares[i]! * 100).toFixed(1)}%` }" />
          <span class="num absolute top-1 text-[10px] text-fg">{{ Math.round(shares[i]! * 100) }}%</span>
        </div>
        <input
          v-model.number="scores[i]"
          type="range"
          min="0"
          max="1"
          step="0.01"
          class="ui-range mt-2 w-full"
          :aria-label="`Score against the past self from ${ages[i]} games`"
          @focus="focus = i"
        >
        <p class="num text-[10px] text-fg-muted">s = {{ s.toFixed(2) }}</p>
        <p class="num text-[10px] text-fg-subtle">{{ ages[i] }}</p>
      </div>
    </div>
    <template #caption>
      Ten past selves, oldest on the left, frozen every 5,000 games. The bars are each one's share of the pool games; the sliders are the
      network's running score against it. Uniformly, every past self gets 10%, including the ones it beats almost every time. Prioritized,
      the games go to where there's something left to learn, and a past self it has <em>forgotten</em> how to beat is pulled straight back in.
    </template>
  </UiFigure>
</template>
