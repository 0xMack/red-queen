<script setup lang="ts">
// A plate: the one panel every page builds with. An optional caption row -- a mono `label` ("Fig. 2",
// "Leaderboard", "Live lab"), a `title`, and `actions` on the right -- sits above a hairline rule; `ticks` adds the
// corner registration marks, for the one or two panels a page is actually about. `pad` is the body's padding;
// "none" for panels whose content brings its own (a table, a board that should run edge to edge). `fill` makes the
// body a flex column that takes the panel's remaining height (for a bounded panel whose list scrolls inside).
withDefaults(
  defineProps<{
    label?: string
    title?: string
    ticks?: boolean
    pad?: "none" | "sm" | "md" | "lg"
    as?: string
    fill?: boolean
  }>(),
  { pad: "md", as: "section" },
)
const PAD = { none: "", sm: "p-3", md: "p-4 sm:p-5", lg: "p-5 sm:p-7" } as const
</script>

<template>
  <component :is="as" class="card relative" :class="[ticks ? 'ticks' : '', fill ? 'flex flex-col' : '']">
    <header
      v-if="label || title || $slots.actions || $slots.header"
      class="flex min-h-11 flex-wrap items-center gap-x-3 gap-y-1 border-b border-line px-4 py-2 sm:px-5"
    >
      <slot name="header">
        <span v-if="label" class="label">{{ label }}</span>
        <h3 v-if="title" class="font-display text-[1.3rem] leading-tight font-normal">{{ title }}</h3>
      </slot>
      <div v-if="$slots.actions" class="ml-auto flex flex-wrap items-center gap-2">
        <slot name="actions" />
      </div>
    </header>
    <div :class="[PAD[pad], fill ? 'flex min-h-0 flex-1 flex-col' : '']">
      <slot />
    </div>
    <footer v-if="$slots.footer" class="shrink-0 border-t border-line px-4 py-2.5 text-xs text-fg-subtle sm:px-5">
      <slot name="footer" />
    </footer>
  </component>
</template>
