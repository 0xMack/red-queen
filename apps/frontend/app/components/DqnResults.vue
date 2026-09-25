<script setup lang="ts">
import type { ArmResult } from "~/components/ArmResults.vue"

// `jobs/rl_experiment.py report --name rl-dqn-v1` (docs/design/0010 Phase 2a): every run's final network on the 200
// held-out games. `ladder`: each rung adds one stabilizer to the one before it and is compared with it (seeds
// paired); the first three rungs have 20 seeds, because a divergence in 1 of 5 couldn't be told from a fluke.
// `observers`: the same DQN on three observations of the same game.
defineProps<{ view: "ladder" | "observers" }>()

const DQN = [27.45, 30.0, 29.72, 28.99, 28.57, 28.44, 26.34, 29.36, 27.27, 28.7, 28.57, 27.95, 27.73, 28.57, 28.53, 28.55, 28.45, 27.64, 29.82, 28.34]

const LADDER: ArmResult[] = [
  {
    key: "naive",
    label: "naive",
    sub: "no replay, no target network",
    scores: [0.36, 27.78, 27.43, 27.84, 26.73, 26.87, 26.52, 26.2, 27.6, 26.82, 26.98, 27.52, 27.27, 27.08, 26.43, 27.96, 26.59, 27.54, 26.43, 27.75],
  },
  {
    key: "replay",
    label: "+ replay",
    sub: "50k transitions, batches of 32",
    scores: [28.57, 28.45, 28.57, 0.04, 28.36, 27.98, 27.86, 27.55, 27.89, 28.23, 28.05, 28.52, 28.42, 27.79, 28.38, 28.43, 28.67, 25.89, 28.11, 27.36],
    versus: { label: "naive", difference: 0.873, p: 0.5003 },
  },
  {
    key: "dqn",
    label: "+ target network",
    sub: "copied every 2,000 steps: DQN",
    scores: DQN,
    versus: { label: "+ replay", difference: 1.793, p: 0.0629 },
    highlight: true,
  },
  {
    key: "double",
    label: "+ Double DQN",
    scores: [28.84, 27.73, 28.5, 27.7, 29.6],
    versus: { label: "DQN", difference: -0.474, p: 0.5625 },
  },
  {
    key: "dueling",
    label: "+ dueling heads",
    scores: [28.1, 29.02, 29.19, 27.61, 28.54],
    versus: { label: "+ double", difference: 0.016, p: 1 },
  },
  {
    key: "n3",
    label: "+ 3-step returns",
    scores: [29.88, 29.11, 29.5, 29.09, 29.22],
    versus: { label: "+ dueling", difference: 0.872, p: 0.0625 },
  },
  {
    key: "per",
    label: "+ prioritized replay",
    scores: [29.03, 28.5, 29.82, 30.2, 30.16],
    versus: { label: "+ 3-step", difference: 0.178, p: 0.6875 },
  },
  {
    key: "long",
    label: "all of it, 5M steps",
    sub: "5× the training",
    scores: [29.28, 30.18, 30.61, 31.3, 30.86],
    versus: { label: "+ prioritized (1M)", difference: 0.905, p: 0.0625 },
  },
]

const OBSERVERS: ArmResult[] = [
  { key: "egocentric", label: "egocentric.v1", sub: "27 inputs: rays, food & tail offsets", scores: DQN, highlight: true },
  {
    key: "features",
    label: "features.v1",
    sub: "11 inputs: the Q-table's",
    scores: [19.75, 17.34, 18.14, 18.62, 20.3],
    versus: { label: "egocentric", difference: -10.114, p: 0.0625 },
  },
  {
    key: "grid",
    label: "grid-flat.v1",
    sub: "100 inputs: every cell, 0-3",
    scores: [0.33, 0.34, 0.42, 0.43, 0.38],
    versus: { label: "egocentric", difference: -28.569, p: 0.0625 },
  },
]

const REFERENCES = [
  { value: 17.89, label: "greedy 17.9", color: "#fbbf24" },
  { value: 19.51, label: "best Q-table 19.5", color: "#4ade80" },
]
</script>

<template>
  <ArmResults
    v-if="view === 'ladder'"
    eyebrow="Held-out Snake score · the DQN stability ladder · egocentric.v1, 1M steps"
    :arms="LADDER"
    :references="REFERENCES"
    :axis="[0, 32]"
    :ticks="[0, 8, 16, 24, 32]"
  >
    Each dot is a separate training run whose final network played the same 200 unseen games; the bar is the mean. Each rung adds one idea to the
    rung above and is compared with it, seeds paired (with five pairs, p = 0.06 is the smallest possible: every pair moved the same way). The two
    dots at zero are the runs that diverged. From <code>jobs/rl_experiment.py report --name rl-dqn-v1</code>.
  </ArmResults>
  <ArmResults
    v-else
    eyebrow="Held-out Snake score · the same DQN, three observations · 1M steps"
    :arms="OBSERVERS"
    :references="REFERENCES"
    :axis="[0, 32]"
    :ticks="[0, 8, 16, 24, 32]"
  >
    Same network (two hidden layers of 64), same settings, same budget; only what the snake sees changes. Evolution scored 0.06-0.17 on
    <code>grid-flat.v1</code> too, and couldn't do better on <code>egocentric.v1</code> than on <code>features.v1</code> (NEAT 35.8 vs. 38.0).
  </ArmResults>
</template>
