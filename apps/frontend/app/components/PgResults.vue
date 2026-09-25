<script setup lang="ts">
import type { ArmResult } from "~/components/ArmResults.vue"

// `jobs/rl_experiment.py report --name rl-pg-v1` (docs/design/0010 Phase 3): every run's final policy on the 200
// held-out games, 2M env steps. `ladder`: each rung adds one idea to the one before (seeds paired). `evolution`: PPO
// on neuroevolution's own network and observation (11 -> 16 -> 3, tanh, features.v1) against neuroevolution's runs
// from `neat-vs-neuro-v1` with the same rng seeds. `observers`: PPO on the two egocentric observations.
defineProps<{ view: "ladder" | "evolution" | "observers" }>()

const PPO = [38.91, 41.37, 48.73, 43.55, 44.08]

const LADDER: ArmResult[] = [
  { key: "reinforce", label: "REINFORCE", sub: "whole-episode returns", scores: [27.04, 31.95, 29.82, 23.93, 30.36] },
  {
    key: "baseline",
    label: "+ baseline",
    sub: "a learned V(s) subtracted",
    scores: [33.38, 34.71, 35.65, 34.01, 36.52],
    versus: { label: "REINFORCE", difference: 6.23, p: 0.0625 },
  },
  {
    key: "a2c",
    label: "A2C",
    sub: "+ bootstrapping, GAE(0.95)",
    scores: [36.46, 32.85, 30.36, 31.52, 35.06],
    versus: { label: "+ baseline", difference: -1.598, p: 0.3125 },
  },
  {
    key: "ppo",
    label: "PPO",
    sub: "+ clipped, reused rollouts",
    scores: PPO,
    versus: { label: "A2C", difference: 10.073, p: 0.0625 },
    highlight: true,
  },
]

const EVOLUTION: ArmResult[] = [
  {
    key: "ppo16",
    label: "PPO",
    sub: "2M steps · 15 s",
    scores: [22.73, 18.29, 19.48, 23.87, 22.21],
    highlight: true,
  },
  {
    key: "tournament",
    label: "neuroevolution",
    sub: "tournament · 16.5M steps · 6.4 min",
    scores: [17.93, 16.19, 19.45, 20.53, 16.675],
    versus: { label: "PPO", difference: -3.16, p: 0.0625 },
  },
  {
    key: "lexicase",
    label: "neuroevolution",
    sub: "lexicase · 15.1M steps · 6.1 min",
    scores: [15.945, 14.035, 17.615, 17.625, 16.585],
    versus: { label: "PPO", difference: -4.96, p: 0.0625 },
  },
]

const OBSERVERS: ArmResult[] = [
  { key: "ego1", label: "egocentric.v1", sub: "27: rays, food, tail", scores: PPO },
  {
    key: "ego2",
    label: "egocentric.v2",
    sub: "33: + reachable space",
    scores: [62.39, 64.13, 63.98, 65.5, 59.16],
    versus: { label: "egocentric.v1", difference: 19.705, p: 0.0625 },
    highlight: true,
  },
  {
    key: "ego2-long",
    label: "egocentric.v2",
    sub: "10M steps (5×)",
    scores: [70.17, 60.9, 67.33, 62.67, 69.75],
    versus: { label: "2M steps", difference: 3.132, p: 0.3125 },
  },
]

const REFERENCES = [
  { value: 17.89, label: "greedy 17.9", color: "#fbbf24" },
  { value: 37.95, label: "NEAT 38.0", color: "#a78bfa" },
]
</script>

<template>
  <ArmResults
    v-if="view === 'ladder'"
    eyebrow="Held-out Snake score · the policy-gradient ladder · egocentric.v1, 2M steps"
    :arms="LADDER"
    :references="REFERENCES"
    :axis="[0, 50]"
    :ticks="[0, 10, 20, 30, 40, 50]"
  >
    Each dot is a separate training run whose final policy played the same 200 unseen games (its most likely move each time); the bar is the
    mean of five. Each rung is compared with the one above it, seeds paired -- with five pairs, p = 0.06 means every pair moved the same way. From
    <code>jobs/rl_experiment.py report --name rl-pg-v1</code>.
  </ArmResults>
  <ArmResults
    v-else-if="view === 'evolution'"
    eyebrow="Held-out Snake score · the same 11 → 16 → 3 network, trained two ways"
    :arms="EVOLUTION"
    :references="[{ value: 17.89, label: 'greedy 17.9', color: '#fbbf24' }]"
    :axis="[10, 26]"
    :ticks="[10, 14, 18, 22, 26]"
  >
    Same architecture (243 weights, tanh), same observation (features.v1), same rng seeds; gradient ascent on the return against evolution's
    selection. Neuroevolution's runs are from <code>jobs/snake_experiment.py</code>'s <code>neat-vs-neuro-v1</code>, compared here seed for seed.
  </ArmResults>
  <ArmResults
    v-else
    eyebrow="Held-out Snake score · PPO, two observations · 2M steps (and 10M)"
    :arms="OBSERVERS"
    :references="REFERENCES"
    :axis="[30, 70]"
    :ticks="[30, 40, 50, 60, 70]"
  >
    The same PPO; the only difference is three numbers per move saying how much of the board each move leaves reachable, and three saying whether
    the snake's own tail still is.
  </ArmResults>
</template>
