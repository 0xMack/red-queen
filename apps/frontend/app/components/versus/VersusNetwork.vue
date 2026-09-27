<script setup lang="ts">
import type { Activations, BotNetwork } from "~/types/versus"

// A bot's network, drawn by the same diagrams Snake's champion uses -- NetworkDiagram for a fixed topology,
// NeatDiagram for an evolved graph (where you can watch hidden nodes exist at all) -- and, once the bot has
// evaluated a position, lit up by activation. Game-agnostic: it takes a `BotNetwork` and the activations the
// session recorded. Fills its parent (which fixes the height).
defineProps<{
  network: BotNetwork | null
  /** The network's values on the position the bot chose to leave behind; null before its first decision. */
  activations: Activations | null
}>()
</script>

<template>
  <div class="relative h-full min-h-0 overflow-hidden">
    <NetworkDiagram
      v-if="network?.kind === 'layered'"
      class="mx-auto"
      :weights="network.weights"
      :layer-sizes="network.layerSizes"
      :activations="Array.isArray(activations) ? activations : null"
      :input-labels="network.inputLabels"
      :output-labels="network.outputLabels"
      fit
    />
    <NeatDiagram
      v-else-if="network?.kind === 'graph'"
      class="mx-auto"
      fit
      :genome="network.genome"
      :activations="activations instanceof Map ? activations : null"
      :input-labels="network.inputLabels ? [...network.inputLabels] : []"
      :output-labels="network.outputLabels ? [...network.outputLabels] : []"
    />
    <p v-else class="flex h-full items-center justify-center rounded-md border border-dashed border-line px-4 text-center text-xs text-fg-subtle">
      No network to draw: this seat's player doesn't evaluate with one.
    </p>
    <p v-if="network && !activations" class="absolute inset-x-0 bottom-0 text-center text-[11px] text-fg-subtle">Lights up on this player's first move.</p>
  </div>
</template>
