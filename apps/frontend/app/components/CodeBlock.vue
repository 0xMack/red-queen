<script setup lang="ts">
import type { CodeLang } from "~/composables/useHighlighter"

const props = defineProps<{ lang: CodeLang; code: string }>()

const html = ref<string | null>(null)
const copied = ref(false)

// Highlighted client-side only, after mount: SSR renders the plain <pre> fallback below (no
// hydration mismatch, since `html` starts null on both sides), then this upgrades it once shiki's
// loaded -- a brief flash of unhighlighted code on first load, traded for not slowing down every
// page's SSR response waiting on the highlighter.
onMounted(async () => {
  const highlighter = await useHighlighter()
  html.value = highlighter.codeToHtml(props.code.trim(), { lang: props.lang, theme: "github-dark" })
})

async function copyCode() {
  try {
    await navigator.clipboard.writeText(props.code.trim())
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 1500)
  } catch {
    // Clipboard access can fail (permissions, insecure context) -- not worth surfacing an error
    // for a copy-to-clipboard convenience button; the code is still right there to select by hand.
  }
}
</script>

<template>
  <div class="overflow-hidden rounded-xl border border-line bg-sunken">
    <div class="flex items-center justify-between border-b border-line bg-surface px-4 py-2">
      <span class="flex items-center gap-2 font-mono text-xs text-fg-subtle">
        <span class="flex gap-1"><span class="size-2 rounded-full bg-queen-500/60" /><span class="size-2 rounded-full bg-gold-400/60" /><span class="size-2 rounded-full bg-life-400/60" /></span>
        {{ lang }}
      </span>
      <button class="text-xs text-fg-subtle transition hover:text-fg" @click="copyCode">
        {{ copied ? "Copied!" : "Copy" }}
      </button>
    </div>
    <div v-if="html" class="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed [&_pre]:!bg-transparent" v-html="html" />
    <pre v-else class="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-fg-muted">{{ code.trim() }}</pre>
  </div>
</template>
