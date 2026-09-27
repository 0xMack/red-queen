<script setup lang="ts">
// Which game to play (docs/design/0011): the scenario, kept in the URL (`?scenario=`) so a view can be linked, with
// its lesson and its trap spelled out under it.
const route = useRoute()
const router = useRouter()
const scenario = computed(() => scenarioById(route.query.scenario as string | undefined))
function pick(id: string) {
  router.replace({ query: { ...route.query, scenario: id === "classic" ? undefined : id } })
}
</script>

<template>
  <div class="flex flex-wrap items-start gap-x-5 gap-y-2">
    <UiSelect
      class="w-52"
      label="Scenario"
      :model-value="scenario.id"
      :options="BANDIT_SCENARIOS.map((s) => ({ value: s.id, label: `${s.title} · ${s.arms} × ${s.budget}` }))"
      @update:model-value="pick"
    />
    <p class="max-w-xl min-w-0 flex-1 pt-6 text-xs leading-relaxed text-fg-subtle">
      <span class="text-fg-muted">{{ scenario.lesson }}</span> {{ scenario.pitfall }}
    </p>
  </div>
</template>
