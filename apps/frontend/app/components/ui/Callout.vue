<script setup lang="ts">
// "finding" is a deliberate nod to this project's own established voice -- results and bugs this
// codebase found by actually running things, not guessed at in advance (see
// docs/CODING_GUIDELINES.md's accumulated lessons). Used throughout the Learn chapters to call
// those out explicitly rather than blending them into ordinary prose.
type Variant = "note" | "warning" | "finding"

const props = withDefaults(defineProps<{ variant?: Variant; title?: string }>(), {
  variant: "note",
})

const STYLES: Record<Variant, { rule: string; label: string; accent: string; mark: string }> = {
  note: { rule: "bg-signal-400", label: "Note", accent: "text-signal-300", mark: "N" },
  warning: { rule: "bg-gold-400", label: "Watch out", accent: "text-gold-300", mark: "!" },
  finding: { rule: "bg-life-400", label: "Found by running it", accent: "text-life-300", mark: "✓" },
}

const style = computed(() => STYLES[props.variant])
</script>

<template>
  <aside class="card relative my-8 overflow-hidden py-4 pr-5 pl-6">
    <span class="absolute inset-y-0 left-0 w-[3px]" :class="style.rule" />
    <p class="flex items-baseline gap-2">
      <span class="label" :class="style.accent">{{ style.label }}</span>
      <span v-if="title" class="text-[15px] font-semibold text-fg">{{ title }}</span>
    </p>
    <div class="mt-1.5 text-[14.5px] leading-relaxed text-fg-muted">
      <slot />
    </div>
  </aside>
</template>
