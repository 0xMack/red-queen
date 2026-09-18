<script setup lang="ts">
import type { RunStatus } from "~/types/telemetry"

const props = defineProps<{ status: RunStatus | string; stale?: boolean }>()

const STYLES: Record<string, string> = {
  running: "border-life-400/30 bg-life-400/10 text-life-300",
  completed: "border-signal-400/30 bg-signal-400/10 text-signal-300",
  failed: "border-queen-400/30 bg-queen-400/10 text-queen-300",
  paused: "border-gold-400/30 bg-gold-400/10 text-gold-300",
}
const style = computed(() =>
  props.stale ? "border-line-strong bg-raised text-fg-subtle" : (STYLES[props.status] ?? "border-line bg-raised text-fg-muted"),
)
</script>

<template>
  <span
    class="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap"
    :class="style"
    :title="stale ? 'Marked running, but no update in over 15 minutes -- the job probably exited without recording a final status.' : undefined"
  >
    <span
      class="size-1.5 rounded-full bg-current"
      :class="status === 'running' && !stale ? 'animate-live-pulse' : ''"
    />
    {{ stale ? "stalled?" : status }}
  </span>
</template>
