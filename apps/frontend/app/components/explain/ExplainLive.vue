<script setup lang="ts">
import type { LiveSpec } from "~/types/explain"

// The panel's live view: a bandit strategy playing a scenario on the real Rust core (`BanditPlayer`, WebAssembly),
// with the scenario -- or, for a scenario's explainer, the strategy -- switchable; or an observer watching a snake.
const props = defineProps<{ spec: LiveSpec }>()

const scenario = ref(props.spec.kind === "bandit" ? props.spec.scenario : "classic")
const strategyId = ref("__spec")
watch(
  () => props.spec,
  (s) => {
    if (s.kind === "bandit") scenario.value = s.scenario
    strategyId.value = "__spec"
  },
)

const strategy = computed(() => {
  if (props.spec.kind !== "bandit") return null
  const picked = BANDIT_STRATEGIES.find((s) => s.id === strategyId.value)
  return picked ? { strategy: picked.strategy, params: picked.params, label: picked.label } : { strategy: props.spec.strategy, params: props.spec.params, label: props.spec.label }
})
const info = computed(() => scenarioById(scenario.value))
// A strategy that can't see the lamp still plays a contextual scenario (that's the lesson); a sequential one needs it.
const observer = computed(() => (props.spec.kind === "bandit" && props.spec.observer && info.value.contexts > 1 ? props.spec.observer : observerFor(info.value, false)))
</script>

<template>
  <div>
    <template v-if="spec.kind === 'bandit' && strategy">
      <div class="mb-4 flex flex-wrap items-end gap-3">
        <UiSelect
          v-if="spec.pickScenario"
          v-model="scenario"
          class="w-52"
          label="Scenario"
          :options="BANDIT_SCENARIOS.map((s) => ({ value: s.id, label: `${s.title} · ${s.arms} × ${s.budget}` }))"
        />
        <UiSelect
          v-if="spec.pickStrategy"
          v-model="strategyId"
          class="w-60"
          label="Strategy"
          :options="[{ value: '__spec', label: spec.label }, ...BANDIT_STRATEGIES.map((s) => ({ value: s.id, label: s.label }))]"
        />
      </div>
      <ClientOnly>
        <BanditPlayer :key="`${scenario}:${strategy.strategy}:${strategy.params}:${observer}`" :scenario="scenario" :strategy="strategy.strategy" :params="strategy.params" :observer="observer" :label="strategy.label" />
      </ClientOnly>
    </template>
    <ClientOnly v-else-if="spec.kind === 'observer'">
      <ObserverLive :id="spec.id" />
    </ClientOnly>
  </div>
</template>
