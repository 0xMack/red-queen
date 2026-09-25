<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { probeDevice } from "~/inference/device"

// Continuous control, live: a Gaussian policy (PPO) learning Reach1D (docs/design/0010 Phase 3b) -- an agent on a line that
// chooses an acceleration in [-1, 1] every step and is rewarded -|distance to the target|. The policy network outputs
// the *mean* of a normal distribution; its spread is one more learned number. The track shows the current mean policy
// driving to a target (drawn at the centre: the observation is the offset from it), the bell curve is the action
// distribution for the state it's in, and the chart is the return of recent training episodes. Neither the Q-table
// (without cutting the action into bins) nor DQN (a value per action) can even represent this policy.

const BUDGET = 200_000
const EVAL_EVERY = 10_000
const SPEEDS = [
  { label: "watch", stepsPerSecond: 5_000 },
  { label: "fast", stepsPerSecond: 400_000 }, // as fast as this device trains (~60k steps/s on a desktop)
] as const
// PPO without the entropy bonus: with it, the bonus holds the spread up and it never visibly narrows (measured: 0.6-0.75
// throughout at 0.01). REINFORCE updates once per two 1,000-step episodes here -- too few updates to watch.
const PARAMS = "hidden=16,rollout_steps=1024,entropy_coef=0,learning_rate=0.001"
const HALF_LN_2PI_E = 0.5 * Math.log(2 * Math.PI * Math.E) // a normal's entropy is log σ + this

const config = reactive({ seed: 0 })
const lab = useRlLab({ speed: SPEEDS[0].stepsPerSecond })
const live = ref<boolean | null>(null)
onMounted(async () => {
  live.value = (await probeDevice()).wasm
})

function train() {
  lab.start({
    algorithm: "ppo",
    env: "reach1d",
    params: PARAMS,
    reward: "shaped",
    seed: config.seed,
    budget: BUDGET,
    evalEvery: EVAL_EVERY,
  })
}
const started = computed(() => lab.status.value !== "idle")

// The track: offset from the target in [-6, 6] (targets start within 5 of the agent).
const W = 460
const TRACK_Y = 40
const x = (offset: number) => 20 + ((Math.max(-6, Math.min(6, offset)) + 6) / 12) * (W - 40)
const offset = computed(() => lab.track.value?.observation[0] ?? null)
const velocity = computed(() => lab.track.value?.observation[1] ?? 0)

// The action distribution: N(mean, std) over [-1.6, 1.6]; actions beyond ±1 are clamped by the game.
const BELL_TOP = 80
const BELL_H = 70
const ax = (a: number) => 20 + ((a + 1.6) / 3.2) * (W - 40)
const bell = computed(() => {
  const v = lab.track.value?.values
  if (!v || v.length < 2) return null
  const [mean, std] = [v[0]!, Math.max(v[1]!, 1e-3)]
  const peak = 1 / (std * Math.sqrt(2 * Math.PI))
  const points = Array.from({ length: 81 }, (_, i) => {
    const a = -1.6 + (3.2 * i) / 80
    const density = Math.exp(-0.5 * ((a - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI))
    return `${ax(a).toFixed(1)},${(BELL_TOP + BELL_H - (density / Math.max(peak, 1)) * BELL_H).toFixed(1)}`
  })
  return { mean, std, path: `M${points.join("L")}` }
})

const returnsChart = computed<{ x: number[]; series: ChartSeries[] } | null>(() => {
  const r = lab.returns.value
  if (r.length < 2) return null
  return { x: r.map((p) => p[0]), series: [{ key: "return", label: "return of recent training episodes", color: "#ff5c7a", values: r.map((p) => p[1]), width: 2 }] }
})
const stdNow = computed(() => {
  const last = lab.points.value.at(-1)?.entropy
  return last === null || last === undefined ? null : Math.exp(last - HALF_LN_2PI_E)
})
const steps = (n: number) => (n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n))
</script>

<template>
  <div class="card not-prose my-6 p-4 sm:p-5" data-reach-lab>
    <p v-if="live === false" class="text-sm text-fg-muted">
      This demo trains live and needs WebAssembly, which this browser can't run. Recorded (PPO, seed 0), it looks like this: the policy's mean soon
      steers to any target and brakes on arrival; over 200,000 steps the return climbs from about -90 to -45, and the spread widens from ±0.74 to
      ±0.91 while the mean is still wrong, then narrows to ±0.26.
    </p>
    <template v-else>
      <div class="flex flex-wrap items-center gap-2">
        <button v-if="lab.status.value === 'running'" class="btn-primary btn-sm" @click="lab.pause">Pause</button>
        <button v-else-if="lab.status.value === 'paused'" class="btn-primary btn-sm" @click="lab.resume">Resume</button>
        <button class="btn-sm" :class="lab.status.value === 'running' || lab.status.value === 'paused' ? 'btn-ghost' : 'btn-primary'" :disabled="live !== true" @click="train">
          {{ started ? "Train again" : "Train" }}
        </button>
        <label class="flex items-center gap-2 text-sm">
          <span class="text-fg-muted">seed</span>
          <select v-model.number="config.seed" class="rounded-md border border-line bg-surface px-2 py-1 text-xs" aria-label="random seed">
            <option v-for="s in 10" :key="s" :value="s - 1">{{ s - 1 }}</option>
          </select>
        </label>
        <div class="ml-auto flex rounded-lg border border-line bg-surface p-0.5" role="group" aria-label="training speed">
          <button
            v-for="s in SPEEDS"
            :key="s.label"
            class="rounded-md px-2.5 py-1 text-xs font-medium transition"
            :class="lab.speed.value === s.stepsPerSecond ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
            @click="lab.speed.value = s.stepsPerSecond"
          >
            {{ s.label }}
          </button>
        </div>
      </div>

      <svg :viewBox="`0 0 ${W} ${BELL_TOP + BELL_H + 26}`" class="mt-3 block h-auto w-full" role="img" aria-label="The agent on its track, and the policy's action distribution">
        <!-- track -->
        <line :x1="x(-6)" :x2="x(6)" :y1="TRACK_Y" :y2="TRACK_Y" stroke="#323a4d" stroke-width="2" />
        <line :x1="x(0)" :x2="x(0)" :y1="TRACK_Y - 16" :y2="TRACK_Y + 16" stroke="#4ade80" stroke-width="2" />
        <text :x="x(0)" :y="TRACK_Y - 20" text-anchor="middle" class="fill-life-300 font-mono text-[9px]">target</text>
        <template v-if="offset !== null">
          <line :x1="x(offset)" :x2="x(offset + velocity * 2)" :y1="TRACK_Y" :y2="TRACK_Y" stroke="#fbbf24" stroke-width="3" />
          <circle :cx="x(offset)" :cy="TRACK_Y" r="8" fill="#ff5c7a" />
          <text :x="x(offset)" :y="TRACK_Y + 24" text-anchor="middle" class="fill-fg-subtle font-mono text-[9px]">
            {{ Math.abs(offset).toFixed(2) }} away · step {{ lab.track.value?.step }}
          </text>
        </template>
        <text v-else :x="W / 2" :y="TRACK_Y + 4" text-anchor="middle" class="fill-fg-subtle text-[11px]">Press Train: the policy starts out pushing at random.</text>

        <!-- action distribution -->
        <text x="20" :y="BELL_TOP - 6" class="fill-fg-subtle font-mono text-[9px]">acceleration the policy would choose (its distribution)</text>
        <rect :x="ax(-1.6)" :y="BELL_TOP" :width="ax(-1) - ax(-1.6)" :height="BELL_H" fill="#6b7489" fill-opacity="0.12" />
        <rect :x="ax(1)" :y="BELL_TOP" :width="ax(1.6) - ax(1)" :height="BELL_H" fill="#6b7489" fill-opacity="0.12" />
        <line :x1="ax(-1.6)" :x2="ax(1.6)" :y1="BELL_TOP + BELL_H" :y2="BELL_TOP + BELL_H" stroke="#323a4d" />
        <text v-for="t in [-1, 0, 1]" :key="t" :x="ax(t)" :y="BELL_TOP + BELL_H + 12" text-anchor="middle" class="fill-fg-subtle font-mono text-[9px]">{{ t }}</text>
        <template v-if="bell">
          <path :d="bell.path" fill="none" stroke="#60a5fa" stroke-width="2" />
          <line :x1="ax(Math.max(-1, Math.min(1, bell.mean)))" :x2="ax(Math.max(-1, Math.min(1, bell.mean)))" :y1="BELL_TOP" :y2="BELL_TOP + BELL_H" stroke="#ff5c7a" stroke-width="2" />
          <text :x="W - 20" :y="BELL_TOP + 10" text-anchor="end" class="fill-fg-subtle font-mono text-[9px]">mean {{ bell.mean.toFixed(2) }} · spread ±{{ bell.std.toFixed(2) }}</text>
        </template>
      </svg>

      <dl class="mt-2 grid grid-cols-3 gap-2 text-xs">
        <div><dt class="text-fg-subtle">env steps</dt><dd class="num text-fg">{{ steps(lab.totalSteps.value) }} / {{ steps(BUDGET) }}</dd></div>
        <div><dt class="text-fg-subtle">episodes</dt><dd class="num text-fg">{{ lab.totalEpisodes.value.toLocaleString() }}</dd></div>
        <div><dt class="text-fg-subtle">spread (σ)</dt><dd class="num text-fg">{{ stdNow === null ? "--" : `±${stdNow.toFixed(2)}` }}</dd></div>
      </dl>
      <LineChart v-if="returnsChart" class="mt-2" :x="returnsChart.x" :series="returnsChart.series" :height="130" x-label="env steps" :format="(v: number) => v.toFixed(0)" />
      <p class="mt-1 text-[11px] text-fg-subtle">Return = the sum of -distance over a 1,000-step episode: 0 would be sitting on the target from the start.</p>
      <p v-if="lab.error.value" class="mt-2 text-xs text-queen-300">{{ lab.error.value }}</p>
    </template>
  </div>
</template>
