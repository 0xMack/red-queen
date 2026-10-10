<script setup lang="ts">
// An evolved bandit strategy (the trainer's bandit_evolve, docs/design/0011) on its run's page: the settings evolution
// arrived at, next to what they mean, and the strategy playing the scenario it evolved on (`BanditPlayer`). The champion
// artifact is `{genome, strategy, params}`.
const props = defineProps<{ runId: string; championRef: string; scenarios?: string[] }>()

const api = useApi()
const champion = ref<{ strategy: string; params: Record<string, number> } | null>(null)
const failed = ref(false)
watch(
  () => props.championRef,
  async (ref) => {
    failed.value = false
    try {
      champion.value = JSON.parse(await api.fetch<string>(`/runs/${props.runId}/artifacts/${ref}`, { responseType: "text" }))
    } catch {
      failed.value = true
    }
  },
  { immediate: true },
)

const MEANING: Record<string, string> = {
  epsilon: "share of pulls on a random machine",
  decay: "pulls over which that exploring fades out",
  alpha: "how far each payout moves an estimate",
  initial: "what an untried machine is assumed to pay",
}
const params = computed(() => (champion.value ? Object.entries(champion.value.params).map(([k, v]) => `${k}=${v}`).join(",") : ""))
const scenario = computed(() => props.scenarios?.[0] ?? "classic")
</script>

<template>
  <div>
    <p v-if="failed" class="text-sm text-fg-subtle">Couldn't load this champion's artifact.</p>
    <template v-else-if="champion">
      <UiKeyValues :entries="Object.entries(champion.params).map(([k, v]) => [`${k} -- ${MEANING[k] ?? ''}`, v] as const)" />
      <p class="label mt-5 mb-3">Playing {{ scenarioById(scenario).title }}</p>
      <ClientOnly>
        <BanditPlayer :scenario="scenario" :strategy="champion.strategy" :params="params" label="evolved" />
      </ClientOnly>
    </template>
  </div>
</template>
