<script setup lang="ts">
import type { ArmResult } from "~/components/ArmResults.vue"

// `jobs/rl_experiment.py report --name rl-tabular-v1` (docs/design/0010 Phase 1a): twelve variants of tabular
// Q-learning on Snake (features.v1), five rng seeds each, 1M environment steps each (q-long: 5M), every run's *final*
// table playing the same 200 held-out games. Each variant is compared with the default (`q-learning`), seeds paired.
const scores: Record<string, number[]> = {
  "q-learning": [16.73, 17.3, 19.09, 18.16, 16.89],
}
const vs = (difference: number, p: number) => ({ label: "Q-learning", difference, p })
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const arm = (key: string, label: string, sub: string, s: number[], p: number): ArmResult => ({
  key,
  label,
  sub,
  scores: s,
  versus: vs(mean(s) - mean(scores["q-learning"]!), p),
})

const ARMS: ArmResult[] = [
  { key: "q-learning", label: "Q-learning", sub: "α 0.1 · γ 0.95 · ε 1 → 0.05 over 100k", scores: scores["q-learning"]!, highlight: true },
  arm("q-optimistic", "optimistic start", "every Q starts at 2 · ε 0.02", [19.51, 20.3, 17.27, 18.69, 21.14], 0.1875),
  arm("q-gamma-0.9", "γ 0.9", "shorter horizon", [17.33, 20.2, 19.32, 19.11, 18.52], 0.0625),
  arm("q-eps-fast", "ε fast", "decays over 20k steps", [18.77, 18.16, 18.57, 18.05, 19.89], 0.25),
  arm("sarsa-n3", "SARSA, 3-step", "on-policy, n = 3", [18.68, 18.83, 19.87, 18.25, 17.3], 0.0625),
  arm("q-sparse", "sparse reward", "food +1 · death −1 · nothing else", [18.38, 17.98, 18.3, 18.8, 17.73], 0.25),
  arm("q-n3", "Q-learning, 3-step", "n = 3", [16.95, 17.4, 19.03, 18.11, 18.31], 0.3125),
  arm("q-long", "5× longer", "5M steps", [17.76, 17.97, 18.32, 16.89, 18.54], 0.6875),
  arm("q-alpha-0.3", "α 0.3", "bigger steps", [17.64, 18.32, 15.14, 18.79, 18.11], 1),
  arm("q-eps-slow", "ε slow", "decays over 500k steps", [17.39, 17.44, 17.48, 16.65, 18.93], 1),
  arm("sarsa", "SARSA", "on-policy, n = 1", [16.49, 17.95, 17.05, 18.67, 17.25], 0.9375),
  arm("q-gamma-0.99", "γ 0.99", "longer horizon", [13.99, 18.49, 17.32, 18.0, 18.95], 0.8125),
]
</script>

<template>
  <ArmResults
    eyebrow="Held-out Snake score · twelve Q-learning variants · five seeds each"
    :arms="ARMS"
    :references="[{ value: 17.89, label: 'greedy 17.89', color: '#fbbf24' }]"
    :axis="[12, 24]"
    :ticks="[12, 15, 18, 21, 24]"
  >
    Each dot is a separate training run (a different rng seed) whose final table played the same 200 games none of them trained on; the bar is the
    mean of five. Every run visited the same 256 states. Each trained in about half a second (the 5M-step arm in 2.6 s) in the Rust core. From
    <code>jobs/rl_experiment.py report --name rl-tabular-v1</code>; the runs are on the <NuxtLink to="/runs">runs page</NuxtLink>, grouped under
    the experiment's name.
  </ArmResults>
</template>
