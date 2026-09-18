<script setup lang="ts">
// "finding" is a deliberate nod to this project's own established voice -- results and bugs this
// codebase found by actually running things, not guessed at in advance (see
// docs/CODING_GUIDELINES.md's accumulated lessons). Used throughout the Learn chapters to call
// those out explicitly rather than blending them into ordinary prose.
type Variant = "note" | "warning" | "finding"

const props = withDefaults(defineProps<{ variant?: Variant; title?: string }>(), {
  variant: "note",
})

const STYLES: Record<Variant, { border: string; bg: string; label: string; accent: string; icon: string }> = {
  note: { border: "border-signal-400/25", bg: "bg-signal-400/[0.06]", label: "Note", accent: "text-signal-300", icon: "i" },
  warning: { border: "border-gold-400/25", bg: "bg-gold-400/[0.06]", label: "Watch out", accent: "text-gold-300", icon: "!" },
  finding: { border: "border-life-400/25", bg: "bg-life-400/[0.06]", label: "Found by running it", accent: "text-life-300", icon: "✓" },
}

const style = computed(() => STYLES[props.variant])
</script>

<template>
  <aside class="my-8 flex gap-4 rounded-xl border px-5 py-4" :class="[style.border, style.bg]">
    <span
      class="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full border border-current font-mono text-xs font-bold"
      :class="style.accent"
    >
      {{ style.icon }}
    </span>
    <div class="min-w-0">
      <p class="font-display text-[15px] font-semibold" :class="style.accent">
        {{ title ?? style.label }}
      </p>
      <div class="mt-1.5 text-sm leading-relaxed text-fg-muted">
        <slot />
      </div>
    </div>
  </aside>
</template>
