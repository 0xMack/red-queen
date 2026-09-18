<script setup lang="ts">
// "finding" is a deliberate nod to this project's own established voice -- results and bugs this
// codebase found by actually running things, not guessed at in advance (see
// docs/CODING_GUIDELINES.md's accumulated lessons). Used throughout the Learn chapters to call
// those out explicitly rather than blending them into ordinary prose.
type Variant = "note" | "warning" | "finding"

const props = withDefaults(defineProps<{ variant?: Variant; title?: string }>(), {
  variant: "note",
})

const STYLES: Record<Variant, { border: string; bg: string; label: string }> = {
  note: { border: "border-blue-200", bg: "bg-blue-50", label: "Note" },
  warning: { border: "border-amber-200", bg: "bg-amber-50", label: "Watch out" },
  finding: { border: "border-emerald-200", bg: "bg-emerald-50", label: "Found by running it" },
}

const style = computed(() => STYLES[props.variant])
</script>

<template>
  <div class="my-6 rounded-lg border px-4 py-3" :class="[style.border, style.bg]">
    <p class="text-xs font-semibold uppercase tracking-wide text-slate-500">
      {{ title ?? style.label }}
    </p>
    <div class="mt-1.5 text-sm leading-relaxed text-slate-700">
      <slot />
    </div>
  </div>
</template>
