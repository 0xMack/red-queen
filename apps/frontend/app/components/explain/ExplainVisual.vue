<script setup lang="ts">
import type { VisualSpec } from "~/types/explain"

// An explainer's picture, whatever its kind: one well, one aspect ratio, so every card looks alike.
defineProps<{ spec: VisualSpec }>()
</script>

<template>
  <div class="well overflow-hidden">
    <StrategyGlyph v-if="spec.kind === 'strategy'" :mode="spec.mode" class="p-1.5" />
    <ScenarioGlyph v-else-if="spec.kind === 'scenario'" :id="spec.id" class="p-1.5" />
    <ObserverGlyph v-else-if="spec.kind === 'observer'" :id="spec.id" class="p-1.5" />
    <MetricGlyph v-else-if="spec.kind === 'metric'" :mode="spec.mode" class="p-1.5" />
    <div v-else-if="spec.kind === 'art'" class="aspect-[2/1]">
      <ChapterArt :kind="spec.art" fit="contain" />
    </div>
  </div>
</template>
