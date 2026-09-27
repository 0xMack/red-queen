<script setup lang="ts">
import type { ExplainResults } from "~/types/explain"

// Results as a ranked list of bars: skill on every scenario, who wins a scenario, the entrants of one algorithm.
// A row that names another explainer opens it.
defineProps<{ results: ExplainResults }>()
defineEmits<{ open: [ref: string] }>()

const TONE = {
  best: { bar: palette.gold400, text: "text-gold-300" },
  bad: { bar: palette.queen500, text: "text-queen-300" },
  self: { bar: palette.queen400, text: "text-queen-200" },
  muted: { bar: palette.lineStrong, text: "text-fg-subtle" },
} as const
</script>

<template>
  <div>
    <ol class="space-y-0.5">
      <li v-for="(row, i) in results.rows" :key="`${row.label}${i}`">
        <component
          :is="row.ref ? 'button' : 'div'"
          :type="row.ref ? 'button' : undefined"
          class="grid w-full grid-cols-[minmax(0,1fr)_7rem_3.2rem] items-center gap-3 rounded-[5px] px-2 py-1 text-left text-xs"
          :class="[row.ref ? 'transition hover:bg-raised' : '', row.tone === 'self' ? 'bg-queen-500/10' : '']"
          @click="row.ref && $emit('open', row.ref)"
        >
          <span class="min-w-0 truncate" :class="row.tone ? TONE[row.tone].text : 'text-fg'">
            {{ row.label }}<span v-if="row.sub" class="ml-1.5 font-mono text-[10px] text-fg-subtle">{{ row.sub }}</span>
          </span>
          <span class="relative h-1.5 rounded-full bg-sunken">
            <span
              class="absolute inset-y-0 left-0 rounded-full"
              :style="{ width: `${Math.max(0, Math.min(1, row.value / results.max)) * 100}%`, background: row.tone ? TONE[row.tone].bar : palette.fgSubtle }"
            />
          </span>
          <span class="num text-right" :class="row.tone ? TONE[row.tone].text : 'text-fg-muted'">{{ row.display }}</span>
        </component>
      </li>
    </ol>
    <p v-if="results.caption" class="mt-2 text-[11px] leading-relaxed text-fg-subtle">{{ results.caption }}</p>
  </div>
</template>
