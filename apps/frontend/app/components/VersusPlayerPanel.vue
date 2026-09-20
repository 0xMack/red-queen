<script setup lang="ts">
// One player's diagnostics, framed: a colour-coded header (who, playing what, and what it is doing now)
// around whatever the body shows (candidate moves, its network). The frame has a fixed height and a border
// that brightens for the side to move, so the two panels can sit side by side and the turn passing changes
// a border, not the layout. Game-agnostic.
defineProps<{
  playerName: string
  strategyLabel?: string
  color?: string
  /** It is this player's turn. */
  active?: boolean
  /** What it is doing: "weighing 7…", "just played", "last move". */
  status?: string
}>()
</script>

<template>
  <section class="flex h-[22rem] flex-col rounded-lg border bg-sunken p-3 transition-colors duration-300" :class="active ? 'border-line-strong shadow-[0_0_0_1px_var(--panel-glow)]' : 'border-line'" :style="{ '--panel-glow': color ?? 'transparent' }">
    <header class="flex shrink-0 items-center justify-between gap-2 pb-2 text-xs">
      <p class="flex min-w-0 items-center gap-2">
        <span class="size-2.5 shrink-0 rounded-full" :class="{ 'animate-pulse': active }" :style="{ background: color ?? '#a0a8ba' }" />
        <span class="truncate font-medium text-fg" :title="strategyLabel ? `${playerName} · ${strategyLabel}` : playerName">
          {{ playerName }}<template v-if="strategyLabel"> · {{ strategyLabel }}</template>
        </span>
      </p>
      <span v-if="status" class="shrink-0 text-[11px]" :class="active ? 'text-queen-300' : 'text-fg-subtle'">{{ status }}</span>
    </header>
    <div class="min-h-0 flex-1"><slot /></div>
  </section>
</template>
