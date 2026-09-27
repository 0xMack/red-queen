<script setup lang="ts">
// A row of small readouts -- label over value -- for a lab's progress, a champion's vitals, a run's config.
// `ruled` separates them with hairlines (the instrument-panel look); `cols` fixes the column count.
import type { UiStat } from "~/types/ui"

withDefaults(defineProps<{ items: readonly UiStat[]; cols?: number; ruled?: boolean }>(), { ruled: true })
const TONE = { default: "text-fg", queen: "text-queen-300", life: "text-life-300", gold: "text-gold-300", signal: "text-signal-300" } as const
</script>

<template>
  <dl
    class="grid overflow-hidden text-xs"
    :class="ruled ? 'gap-px rounded-[8px] border border-line bg-line' : 'gap-x-4 gap-y-3'"
    :style="{ gridTemplateColumns: `repeat(${cols ?? items.length}, minmax(0, 1fr))` }"
  >
    <div v-for="s in items" :key="s.label" class="min-w-0" :class="[ruled ? 'bg-surface px-3 py-2' : '', s.wide ? 'col-span-2' : '']">
      <dt class="label truncate" :class="s.label.length <= 2 ? 'normal-case' : ''">{{ s.label }}</dt>
      <dd class="num mt-0.5 truncate text-[13px]" :class="TONE[s.tone ?? 'default']">{{ s.value }}</dd>
    </div>
  </dl>
</template>
