<script setup lang="ts">
// The bandit chapter's live lab (docs/design/0011): pick a strategy and a scenario, and watch it play -- the table it
// keeps, every pull, its beliefs, the reveal. `scenarios` limits the choice (a section about one lesson), `lamp` adds
// the "it can see the lamp" switch that turns one row into two.
const props = withDefaults(
  defineProps<{ title: string; scenarios?: string[]; strategy?: string; lamp?: boolean; initialLamp?: boolean; gamma?: boolean }>(),
  { scenarios: () => BANDIT_SCENARIOS.map((s) => s.id), strategy: "greedy", lamp: false, initialLamp: false, gamma: false },
)
const config = reactive({ strategy: props.strategy, scenario: props.scenarios[0]!, seesLamp: props.initialLamp, gamma: 0.9 })
const info = computed(() => scenarioById(config.scenario))
const observer = computed(() => observerFor(info.value, config.seesLamp))
// `gamma`: the Q-learning agent alone, with its discount on a dial (0 = only the next payout counts).
const GAMMAS = [0, 0.5, 0.8, 0.9, 0.99]
const chosen = computed(() =>
  props.gamma
    ? { id: "q", strategy: "q_table", params: `gamma=${config.gamma},initial_q=10,alpha=0.5,epsilon=0`, label: `Q-learning, γ ${config.gamma}` }
    : (BANDIT_STRATEGIES.find((s) => s.id === config.strategy) ?? BANDIT_STRATEGIES[0]!),
)
</script>

<template>
  <LabFrame :live="true" :title="title" split="none" data-bandit-lab>
    <div class="flex flex-wrap items-end gap-x-5 gap-y-3">
      <UiSelect v-if="!gamma" v-model="config.strategy" class="w-56" label="Strategy" :options="BANDIT_STRATEGIES.map((s) => ({ value: s.id, label: s.label }))" />
      <UiSelect
        v-else
        v-model="config.gamma"
        class="w-56"
        label="γ, how much the future counts"
        :options="GAMMAS.map((g) => ({ value: g, label: g === 0 ? '0 -- only the next payout' : String(g) }))"
      />
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
      <BanditPlayer class="mt-5" :scenario="config.scenario" :strategy="chosen.strategy" :params="chosen.params" :observer="observer" :label="chosen.label" />
    </ClientOnly>
  </LabFrame>
</template>
