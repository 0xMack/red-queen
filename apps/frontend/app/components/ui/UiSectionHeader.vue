<script setup lang="ts">
import type { RouteLocationRaw } from "vue-router"

// Every section and page opens the same way: a mono eyebrow (optionally numbered, "02 / Leaderboard"), a serif
// title, a lede in the default slot, and an optional link or controls on the right. `level: 1` makes it the page's
// h1, at display size; `back` puts a breadcrumb to the parent page in front of the eyebrow.
const props = withDefaults(
  defineProps<{
    eyebrow?: string
    index?: number
    title: string
    level?: 1 | 2
    to?: RouteLocationRaw
    linkLabel?: string
    live?: boolean
    back?: { to: RouteLocationRaw; label: string }
  }>(),
  { level: 2 },
)
const tag = computed(() => (props.level === 1 ? "h1" : "h2"))
</script>

<template>
  <div class="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
    <div class="max-w-3xl min-w-0">
      <p v-if="eyebrow || index !== undefined || back" class="eyebrow flex items-center gap-2">
        <template v-if="back">
          <NuxtLink :to="back.to" class="text-fg-subtle transition hover:text-fg">← {{ back.label }}</NuxtLink>
          <span class="text-fg-subtle/60">/</span>
        </template>
        <span v-if="live" class="size-1.5 animate-live-pulse rounded-full bg-queen-400" />
        <span v-if="index !== undefined" class="text-fg-subtle">{{ String(index).padStart(2, "0") }} /</span>
        {{ eyebrow }}
      </p>
      <component
        :is="tag"
        class="mt-2.5"
        :class="level === 1 ? 'text-[2.6rem] leading-[1.02] sm:text-[3.4rem]' : 'text-[2rem] leading-[1.08] sm:text-[2.35rem]'"
      >
        <slot name="title">{{ title }}</slot>
      </component>
      <div v-if="$slots.default" class="mt-3 text-[15px] leading-relaxed text-fg-muted" :class="level === 1 ? 'text-base sm:text-lg' : ''">
        <slot />
      </div>
    </div>
    <div v-if="$slots.actions || to" class="flex shrink-0 flex-wrap items-center gap-2">
      <slot name="actions">
        <NuxtLink :to="to!" class="group inline-flex items-center gap-1.5 text-sm text-fg-muted transition hover:text-fg">
          {{ linkLabel }} <span class="transition group-hover:translate-x-0.5">→</span>
        </NuxtLink>
      </slot>
    </div>
  </div>
</template>
