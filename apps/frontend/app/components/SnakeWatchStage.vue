<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { useModelCatalog } from "~/composables/useModelCatalog"

// Snake's Watch stage for the shared game page: the selected entrant playing, in a Web Worker -- a
// published champion as an ONNX model package (or exported on demand), a baseline from the WASM core.
// The page keys this by entrant, so switching remounts it (the session worker itself stays warm).
// Props are spelled out (the shape of `StageProps` in games/types.ts): the SFC compiler can't resolve an
// imported type used as the whole props type without TypeScript installed.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()
const models = props.device.context as ReturnType<typeof useModelCatalog>
</script>

<template>
  <WatchChampion
    v-if="entry"
    :run-id="entry.run_id ?? undefined"
    :packaged="models.availability.value[entry.entrant_id] ?? null"
    :packaged-loading="!models.catalog.value && !models.error.value"
    :baseline="
      baselineName(entry)
        ? { name: baselineName(entry)!, interface: entry.interface, description: entry.metrics.model.description }
        : undefined
    "
    @model-failed="(f) => models.reportFailure(f.packageId, f.variantId, f.backend as 'wasm' | 'webgpu', f.message)"
    @retry-failed="models.retryFailed"
  />
</template>
