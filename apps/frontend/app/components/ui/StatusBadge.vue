<script setup lang="ts">
import type { RunStatus } from "~/types/telemetry"

// A run's status as a badge. `stale`: marked running, but silent too long to still be alive.
const props = defineProps<{ status: RunStatus | string; stale?: boolean }>()
const TONES = { running: "life", completed: "signal", failed: "queen", paused: "gold" } as const
const tone = computed(() => (props.stale ? "neutral" : (TONES[props.status as keyof typeof TONES] ?? "neutral")))
</script>

<template>
  <UiBadge
    :tone="tone"
    dot
    :pulse="status === 'running' && !stale"
    :title="stale ? 'Marked running, but no update in over 15 minutes -- the job probably exited without recording a final status.' : undefined"
  >
    {{ stale ? "stalled?" : status }}
  </UiBadge>
</template>
