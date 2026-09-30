<script setup lang="ts">
import type { BanditUpdate } from "~/composables/useBanditRun"
import * as F from "~/data/math/multi-armed-bandits"
import type { BoundValue, NumFormat } from "~/utils/math/expr"

// The bandit chapter's live lab (docs/design/0011): pick a strategy and a scenario, and watch it play -- the table it
// keeps, every pull, its beliefs, the reveal. `scenarios` limits the choice (a section about one lesson), `lamp` adds
// the "it can see the lamp" switch that turns one row into two.
//
// `math` puts the section's formula on top (docs/design/0012), in one term scope with the lab: every pull fills the
// formula's worked line with the real update the Rust strategy just made, the table and the tape light up with the
// terms, the formula's knobs (ε, α, c, γ) are sliders linked to their symbols, and pinning a term holds the game still.
const props = withDefaults(
  defineProps<{
    title: string
    scenarios?: string[]
    strategy?: string
    lamp?: boolean
    initialLamp?: boolean
    gamma?: boolean
    math?: "incremental" | "constant-step" | "epsilon" | "ucb" | "bellman" | null
    /** Show the formula in the lab; off when the chapter shows it in the prose, in the same <MathScope>. */
    showFormula?: boolean
  }>(),
  { scenarios: () => BANDIT_SCENARIOS.map((s) => s.id), strategy: "greedy", lamp: false, initialLamp: false, gamma: false, math: null, showFormula: true },
)
const config = reactive({ strategy: props.strategy, scenario: props.scenarios[0]!, seesLamp: props.initialLamp, gamma: 0.9 })
const knobs = reactive({ epsilon: 0.1, alpha: 0.2, c: 1.41 })
// A knob restarts the game with its new setting once the slider settles, not on every step of a drag.
const applied = reactive({ ...knobs })
let settle: ReturnType<typeof setTimeout> | null = null
watch(knobs, () => {
  if (settle) clearTimeout(settle)
  settle = setTimeout(() => Object.assign(applied, knobs), 350)
})
const info = computed(() => scenarioById(config.scenario))
const observer = computed(() => observerFor(info.value, config.seesLamp))
// `gamma`: the Q-learning agent alone, with its discount on a dial (0 = only the next payout counts).
const GAMMAS = [0, 0.5, 0.8, 0.9, 0.99]
const Q_ALPHA = 0.5
const Q_INITIAL = 10
const chosen = computed(() => {
  if (props.gamma) {
    return { id: "q", strategy: "q_table", params: `gamma=${config.gamma},initial_q=${Q_INITIAL},alpha=${Q_ALPHA},epsilon=0`, label: `Q-learning, γ ${config.gamma}` }
  }
  // A formula's knob overrides the strategy's own setting, so the slider is the setting.
  if (props.math === "epsilon") return { id: "e", strategy: "epsilon_greedy", params: `epsilon=${applied.epsilon}`, label: `ε-greedy (ε ${applied.epsilon})` }
  if (props.math === "constant-step") return { id: "t", strategy: "epsilon_greedy", params: `epsilon=0.1,alpha=${applied.alpha}`, label: `ε-greedy, step ${applied.alpha}` }
  if (props.math === "ucb") return { id: "u", strategy: "ucb1", params: `c=${applied.c}`, label: `UCB (c ${applied.c})` }
  return BANDIT_STRATEGIES.find((s) => s.id === config.strategy) ?? BANDIT_STRATEGIES[0]!
})

// --- The formula ------------------------------------------------------------------------------------------------------
// The chapter's scope when it wraps the lab in a <MathScope> (prose terms then link to the lab too), else its own.
const scope = props.math ? useOrProvideTermScope() : useTermScope()
const formula = computed(() => {
  switch (props.math) {
    case "incremental":
    case "constant-step":
      return F.incremental
    case "epsilon":
      return F.epsilonGreedy
    case "ucb":
      return F.ucb
    case "bellman":
      return F.bellman
    default:
      return null
  }
})
const forms = computed<Record<string, string>>(() => (props.math === "constant-step" ? { step: "alpha" } : ({} as Record<string, string>)))
const last = shallowRef<BanditUpdate | null>(null)

// The last pull, as the formula's *inputs* -- read from the real update. The formula works out the rest itself (the
// error, the TD term, the bonus, the score, the new value) and, in development, checks its result against `expected`:
// what the Rust strategy actually reported. A disagreement is a bug in the formula, the lab's reading, or the strategy.
const values = computed<Record<string, BoundValue | undefined>>(() => {
  const u = last.value
  const knobValues = { epsilon: knobs.epsilon, c: knobs.c, alpha: props.math === "bellman" ? Q_ALPHA : knobs.alpha, gamma: config.gamma }
  if (!u) return props.math === "bellman" ? { gamma: config.gamma, alpha: Q_ALPHA } : knobValues
  // The worked row is the game's real update, made with the settings the game was started with.
  const played = { ...knobValues, epsilon: applied.epsilon, c: applied.c, alpha: props.math === "bellman" ? Q_ALPHA : applied.alpha }
  switch (props.math) {
    case "incremental":
    case "constant-step":
      return { ...played, "q-old": u.before, n: u.n, reward: u.reward }
    case "ucb": {
      const b = u.chosenFrom
      const n = b?.counts[u.arm] ?? 0
      if (!b || n === 0) return knobValues // an untried machine is pulled before any score is compared
      const t = b.counts.reduce((s, x) => s + x, 0)
      // the machine's name, as wide as the widest letter whichever machine it is
      return { ...played, arm: `\\mathrlap{\\text{${armName(u.arm)}}}\\phantom{\\text{M}}`, estimate: b.values[u.arm], count: n, t }
    }
    case "bellman":
      return { gamma: config.gamma, alpha: Q_ALPHA, "q-old": u.before, reward: u.reward, "max-next": u.nextMax }
    default:
      return knobValues
  }
})
const expected = computed((): Record<string, number> => {
  const u = last.value
  if (!u) return {}
  if (props.math === "incremental" || props.math === "constant-step" || props.math === "bellman") return { "q-new": u.after }
  if (props.math === "ucb" && u.chosenFrom && u.chosenFrom.counts[u.arm]! > 0) {
    // UCB's reported spread is c · scale · √(ln t / N); the formula shows scale 1 -- true for every win-or-lose scenario
    return u.game.rewardScale === 1 ? { bonus: u.chosenFrom.spread[u.arm]! } : {}
  }
  return {}
})
watch(expected, (e) => (scope.expected.value = e), { immediate: true })
watch(values, (v) => (scope.values.value = v), { immediate: true })

// How wide each number can get in this game, so the worked row keeps its width from pull to pull (NumFormat): counts
// up to the budget, payouts up to the scenario's largest, estimates up to that or the optimistic start -- or, looking
// ahead, the most a discounted future can add up to: r_max / (1 − γ).
const formats = computed((): Record<string, NumFormat> => {
  const g = last.value?.game
  if (!g) return {}
  const binary = info.value.binary
  // A minus sign is reserved only where a value can really go negative: Gaussian payouts (and so the estimates).
  const negative = g.minPayout < 0
  const initial = props.math === "bellman" ? Q_INITIAL : 0
  const qMax = Math.max(initial, props.math === "bellman" && config.gamma < 1 ? g.maxPayout / (1 - config.gamma) : g.maxPayout)
  const bonusMax = 3 * g.rewardScale * Math.sqrt(Math.log(g.budget))
  const value = { decimals: 2, max: qMax, signed: negative }
  const difference = { decimals: 2, max: 2 * qMax, signed: true }
  const count = { decimals: 0, max: g.budget }
  return {
    n: count,
    t: count,
    count,
    reward: { decimals: binary ? 0 : 2, max: g.maxPayout, signed: negative },
    "q-old": value,
    "q-new": value,
    estimate: value,
    "max-next": value,
    error: difference,
    td: difference,
    bonus: { decimals: 2, max: bonusMax },
    score: { decimals: 2, max: qMax + bonusMax, signed: negative },
  }
})
watch(formats, (f) => (scope.formats.value = f), { immediate: true })
// ε-greedy: light the branch each pull took, in the formula and the narration.
watch(last, (u) => {
  if (u && props.math === "epsilon") scope.flash(u.greedy ? "exploit" : "explore", 450)
})
const hold = computed(() => scope.pinned.value !== null)

// What the last pull was, in words, under the formula: the pull number, the machine (in its colour), and what happened.
const narration = computed(() => {
  const u = last.value
  if (!u) return null
  const base = { pull: u.pull, arm: u.arm, machine: armName(u.arm), paid: u.reward % 1 === 0 ? String(u.reward) : u.reward.toFixed(2), term: null as string | null }
  if (props.math === "epsilon") return { ...base, term: u.greedy ? "exploit" : "explore", text: u.greedy ? "it had the best estimate" : "not the best estimate: an exploring pull" }
  if (props.math === "ucb" && (u.chosenFrom?.counts[u.arm] ?? 0) === 0) return { ...base, text: "never pulled before, so it goes first: its bonus is infinite" }
  return { ...base, text: null }
})
</script>

<template>
  <LabFrame :live="true" :title="title" split="none" data-bandit-lab>
    <div v-if="formula" class="math-panel mb-5">
      <MathFormula v-if="showFormula" :formula="formula" :forms="forms" bare />
      <p class="mt-2 flex min-h-5 flex-wrap items-center justify-center gap-x-2 gap-y-1 border-t border-line pt-2.5 text-xs text-fg-subtle">
        <template v-if="narration">
          <!-- fixed widths (tabular digits, room for the largest), so the centred line doesn't shift from pull to pull -->
          <span class="num inline-block min-w-[8ch] text-right text-fg-subtle">pull {{ narration.pull }}</span>
          <span class="flex items-center gap-1.5 text-fg">
            <span class="size-2 rounded-full" :style="{ background: armColor(narration.arm) }" />machine {{ narration.machine }}
          </span>
          <span>paid <span class="num inline-block min-w-[5ch] text-left text-fg" v-bind="scope.target('reward')">{{ narration.paid }}</span></span>
          <template v-if="narration.term">
            · <span class="rounded px-1 text-fg-muted" v-bind="scope.target(narration.term)">{{ narration.term }}</span>
          </template>
          <span v-if="narration.text">· {{ narration.text }}</span>
        </template>
        <span v-else>Waiting for the first pull…</span>
        <span v-if="hold" class="ml-1 rounded-full border border-line-strong px-2 py-px text-[10.5px] text-fg-muted">paused while a term is pinned</span>
      </p>
    </div>

    <div class="flex flex-wrap items-end gap-x-5 gap-y-3">
      <UiSelect v-if="!gamma && !math" v-model="config.strategy" class="w-56" label="Strategy" :options="BANDIT_STRATEGIES.map((s) => ({ value: s.id, label: s.label }))" />
      <div v-else-if="gamma" v-bind="scope.target('gamma')" class="w-56">
        <UiSelect
          v-model="config.gamma"
          label="γ, how much the future counts"
          :options="GAMMAS.map((g) => ({ value: g, label: g === 0 ? '0 -- only the next payout' : String(g) }))"
        />
      </div>
      <div v-if="math === 'epsilon'" v-bind="scope.target('epsilon')" class="w-56">
        <UiRange v-model="knobs.epsilon" label="ε, how often it explores" :min="0" :max="0.5" :step="0.01" />
      </div>
      <div v-if="math === 'constant-step'" v-bind="scope.target('alpha')" class="w-56">
        <UiRange v-model="knobs.alpha" label="α, the step" :min="0.02" :max="1" :step="0.02" />
      </div>
      <div v-if="math === 'ucb'" v-bind="scope.target('c')" class="w-56">
        <UiRange v-model="knobs.c" label="c, the exploration weight" :min="0" :max="3" :step="0.01" />
      </div>
      <UiSelect
        v-if="scenarios.length > 1"
        v-model="config.scenario"
        class="w-52"
        label="Scenario"
        :options="scenarios.map((id) => ({ value: id, label: scenarioById(id).title }))"
      />
      <UiCheck v-if="lamp && info.contexts > 1 && !info.sequential" v-model="config.seesLamp" class="pb-1.5">It can see the lamp</UiCheck>
    </div>
    <p class="mt-2 text-xs text-fg-subtle"><span class="text-fg-muted">{{ info.lesson }}</span> {{ info.pitfall }}</p>
    <ClientOnly>
      <BanditPlayer
        class="mt-5"
        :scenario="config.scenario"
        :strategy="chosen.strategy"
        :params="chosen.params"
        :observer="observer"
        :label="chosen.label"
        :hold="hold"
        @update="(u) => (last = u)"
      />
    </ClientOnly>
  </LabFrame>
</template>
