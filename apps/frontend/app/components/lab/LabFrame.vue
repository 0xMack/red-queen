<script setup lang="ts">
// The frame every live lab in the Learn chapters shares: a plate captioned "Live lab" with what's being trained and
// where (your browser, or a recording), a stage on the left (the policy playing), the controls and curves on the
// right, and anything wide below. Without WebAssembly (`live === false`) the `recorded` slot replaces all of it.
// Labs differ only in what they put in the slots -- see QLearningLab, DqnLab, PolicyGradientLab, ReachLab.
withDefaults(defineProps<{ live: boolean | null; title: string; split?: "stage" | "none"; runtime?: string }>(), { split: "stage", runtime: "WebAssembly" })
</script>

<template>
  <UiPanel as="figure" class="ui-figure not-prose my-8" pad="md">
    <template #header>
      <span class="flex items-center gap-2">
        <span class="size-1.5 rounded-full" :class="live === false ? 'bg-fg-subtle' : 'animate-live-pulse bg-queen-400'" />
        <span class="fig-num label text-queen-300">Live lab</span>
      </span>
      <span class="text-sm font-semibold">{{ title }}</span>
      <UiBadge class="ml-auto" :tone="live === false ? 'gold' : 'neutral'">
        {{ live === null ? "checking this device…" : live ? `runs in your browser · ${runtime}` : "recorded run" }}
      </UiBadge>
    </template>

    <div v-if="live === false">
      <slot name="recorded" />
    </div>
    <template v-else>
      <div v-if="split === 'stage'" class="grid gap-6 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <div class="min-w-0"><slot name="stage" /></div>
        <div class="min-w-0"><slot /></div>
      </div>
      <slot v-else />
      <div v-if="$slots.below" class="mt-6 border-t border-line pt-5">
        <slot name="below" />
      </div>
    </template>
  </UiPanel>
</template>
