<script setup lang="ts">
import type { ResolvedExplainer } from "~/types/explain"

// The ⓘ popover's content, for any kind: what it is in two sentences, one picture, a few facts, and the way on --
// a link or two, and "More" (the side panel with everything). Also usable inline, wherever a compact explainer fits.
defineProps<{ explainer: ResolvedExplainer }>()
defineEmits<{ more: []; navigate: [] }>()

const KIND_LABEL = { algorithm: "Algorithm", entrant: "Entrant", scenario: "Scenario", representation: "Representation", metric: "Measure" } as const
</script>

<template>
  <article class="text-left">
    <header>
      <p class="label flex items-center gap-2">
        <span class="text-queen-300">{{ KIND_LABEL[explainer.kind] }}</span>
        <span class="truncate normal-case tracking-normal">{{ explainer.eyebrow }}</span>
      </p>
      <h3 class="mt-1.5 font-display text-[1.45rem] leading-[1.1] font-normal text-fg">{{ explainer.title }}</h3>
    </header>
    <p class="mt-2 text-[13px] leading-relaxed text-fg-muted">{{ explainer.summary }}</p>
    <ExplainVisual v-if="explainer.visual" :spec="explainer.visual" class="mt-3" />
    <ExplainFacts v-if="explainer.facts.length" :facts="explainer.facts.slice(0, 4)" class="mt-3" />
    <footer class="mt-3.5 flex items-end justify-between gap-3 border-t border-line pt-3">
      <ExplainLinks :links="explainer.links.slice(0, 2)" class="min-w-0" @navigate="$emit('navigate')" />
      <button type="button" class="btn-primary btn-sm shrink-0" @click="$emit('more')">More <span aria-hidden="true">→</span></button>
    </footer>
  </article>
</template>
