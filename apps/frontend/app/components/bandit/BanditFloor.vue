<script setup lang="ts">
import type { BanditRunState } from "~/composables/useBanditRun"
import type { BanditScenario } from "~/utils/bandit"

// The casino floor for one run (docs/design/0011): every machine, the lamp (in a scenario that has one), how many
// pulls are left and what has been won. Draws a `useBanditRun` state; `interactive` lets a person pull, `beliefs`
// overlays what a strategy thinks each machine pays, `revealed` shows the truth after the game.
const props = withDefaults(
  defineProps<{
    run: BanditRunState
    scenario: BanditScenario
    interactive?: boolean
    /** The machine a strategy is about to pull. */
    chosen?: number | null
    /** Show the strategy's beliefs for the row it's in. */
    showBeliefs?: boolean
    spinMs?: number
    /** Show the truth (after the game). */
    revealed?: boolean
    /** Hide the running skill (a person's own game: it would give the best machine away). */
    hideSkill?: boolean
  }>(),
  { interactive: false, chosen: null, showBeliefs: false, spinMs: 360, revealed: false, hideSkill: false },
)
const emit = defineEmits<{ pull: [arm: number] }>()

const many = computed(() => props.run.arms.value > 8)

// Per-machine tallies from the run's history (the core counts pulls; wins and sums are the page's to show).
const tallies = computed(() => {
  const n = props.run.arms.value
  const t = Array.from({ length: n }, () => ({ pulls: 0, last: null as number | null, spin: 0, recent: [] as number[] }))
  for (const p of props.run.history.value) {
    const m = t[p.arm]!
    m.pulls += 1
    m.last = p.reward
    m.spin += 1
    m.recent.push(p.reward)
  }
  for (const m of t) m.recent = m.recent.slice(-(many.value ? 5 : 8))
  return t
})

const row = computed(() => (props.run.spec.value?.observer === "lamp.v1" ? props.run.lamp.value : 0))
const beliefs = computed(() => (props.showBeliefs ? (props.run.beliefs.value[row.value] ?? null) : null))
// Bars share one scale: the largest payout a machine of this scenario can make on average, give or take.
const scale = computed(() => (props.scenario.binary ? 1 : props.scenario.id === "jackpot" ? 1.5 : 10))

// The truth, for the reveal: under the lamp lit now, after any drift (what "best" means at the end).
const truth = computed(() => (props.revealed ? props.run.means() : null))
const best = computed(() => (props.revealed ? props.run.bestArm() : -1))

</script>

<template>
  <div>
    <!-- Budget, payout, lamp -->
    <div class="flex flex-wrap items-center gap-x-6 gap-y-2">
      <div class="min-w-40 flex-1">
        <div class="flex items-baseline justify-between font-mono text-[11px] text-fg-subtle">
          <span class="label">pulls</span>
          <span><span class="text-fg">{{ run.pulls.value }}</span> / {{ run.budget.value }}</span>
        </div>
        <div class="mt-1 h-[3px] overflow-hidden rounded-full bg-raised">
          <div class="h-full rounded-full bg-fg-muted transition-[width] duration-200" :style="{ width: `${(run.pulls.value / Math.max(1, run.budget.value)) * 100}%` }" />
        </div>
      </div>
      <div class="text-right">
        <p class="label">won</p>
        <p class="num text-xl leading-tight">{{ scenario.binary ? run.total.value : run.total.value.toFixed(1) }}</p>
      </div>
      <div v-if="!hideSkill || run.done.value" class="text-right">
        <p class="label">skill</p>
        <p class="num text-xl leading-tight" :class="run.skill.value > 0.5 ? 'text-life-300' : run.skill.value > 0.15 ? 'text-fg' : 'text-queen-300'">
          {{ run.pulls.value ? Math.round(run.skill.value * 100) : "--" }}
        </p>
      </div>
      <div v-if="scenario.contexts > 1" class="flex items-center gap-2" aria-live="polite">
        <span class="label">lamp</span>
        <span
          class="lamp size-6 rounded-full transition-colors duration-150"
          :class="run.lamp.value === 0 ? 'lamp-red' : 'lamp-blue'"
          :title="run.lamp.value === 0 ? 'the lamp is red' : 'the lamp is blue'"
        />
      </div>
    </div>

    <!-- The machines -->
    <div class="mt-4 grid gap-2.5" :class="many ? 'grid-cols-4 sm:grid-cols-8' : 'grid-cols-5'">
      <SlotMachine
        v-for="(m, arm) in tallies"
        :key="arm"
        :arm="arm"
        :spin="m.spin"
        :reward="m.last"
        :binary="scenario.binary"
        :pulls="m.pulls"
        :recent="m.recent"
        :spin-ms="spinMs"
        :interactive="interactive && !run.done.value"
        :chosen="chosen === arm"
        :estimate="beliefs ? beliefs.values[arm] : null"
        :spread="beliefs ? beliefs.spread[arm] : null"
        :probability="beliefs && beliefs.probabilities.length ? beliefs.probabilities[arm] : null"
        :scale="scale"
        :true-mean="truth ? truth[arm] : null"
        :best="best === arm"
        :compact="many"
        @pull="(a) => emit('pull', a)"
      />
    </div>
  </div>
</template>

<style scoped>
.lamp {
  border: 1px solid var(--color-line-strong);
}
.lamp-red {
  background: radial-gradient(circle at 35% 35%, #ffb3a6, var(--color-queen-500));
  box-shadow: 0 0 16px 2px rgb(239 70 48 / 0.55);
}
.lamp-blue {
  background: radial-gradient(circle at 35% 35%, #d4e2ff, var(--color-signal-400));
  box-shadow: 0 0 16px 2px rgb(128 169 255 / 0.55);
}
</style>
