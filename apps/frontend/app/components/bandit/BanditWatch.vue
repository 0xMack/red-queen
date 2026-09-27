<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"

// The bandit's Watch stage (docs/design/0011): the leaderboard entrant on stage plays game after game of the chosen
// scenario (`BanditPlayer`), with the settings it was ranked with. In `two-lamps` it can be allowed to see the lamp
// (a row per colour) or not (one row): the contextual lesson, on demand.
// Props spelled out (StageProps, games/types.ts): the SFC compiler can't resolve an imported type as the whole props.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()

const route = useRoute()
const scenario = computed(() => scenarioById(route.query.scenario as string | undefined))
const strategy = computed(() => (props.entry ? entrantStrategy(props.entry) : null) ?? { strategy: "thompson", params: "" })
const seesLamp = ref(true)
const observer = computed(() => observerFor(scenario.value, seesLamp.value))
</script>

<template>
  <div class="space-y-5">
    <div class="flex flex-wrap items-end justify-between gap-4">
      <BanditScenarioPicker class="flex-1" />
      <UiCheck v-if="scenario.contexts > 1 && !scenario.sequential" v-model="seesLamp">It can see the lamp</UiCheck>
    </div>
    <BanditPlayer :scenario="scenario.id" :strategy="strategy.strategy" :params="strategy.params" :observer="observer" :label="entry?.label" />
  </div>
</template>
