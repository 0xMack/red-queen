<script setup lang="ts">
import { CODE_LANG_LABELS, CODE_LANGS, type CodeLang, type CodeVariant, type Snippet } from "~/types/code"

// A highlighted snippet, in one language (`lang` + `code`) or several (`snippet`). With several, the header is a tab
// per language and the choice is the reader's site-wide preference (useCodeLanguage): pick Rust here and every block
// shows Rust, on this page and the next. A block without the preferred language shows its first one in
// pseudocode → Python → Rust order. The header says where the code comes from: a repo path when it's the project's
// own, "translation" when it was written for the page from another language's real code.
const props = defineProps<{ snippet?: Snippet; lang?: CodeLang; code?: string }>()

const variants = computed<{ lang: CodeLang; variant: CodeVariant }[]>(() => {
  const snippet: Snippet = props.snippet ?? (props.lang && props.code !== undefined ? { [props.lang]: props.code } : {})
  return CODE_LANGS.filter((l) => snippet[l] !== undefined).map((l) => {
    const v = snippet[l]!
    return { lang: l, variant: typeof v === "string" ? { code: v } : v }
  })
})
const hasRealSource = computed(() => variants.value.some((v) => v.variant.source))

const { preferred, setPreferred } = useCodeLanguage()
const active = computed(() => variants.value.find((v) => v.lang === preferred.value) ?? variants.value[0]!)
const code = computed(() => active.value.variant.code.trim())

// Highlighted client-side only, after mount: SSR renders the plain <pre> fallback below (no hydration mismatch,
// since nothing is highlighted on either side at first), then this upgrades it once shiki's loaded -- a brief flash
// of unhighlighted code on first load, traded for not slowing down every page's SSR response waiting on the
// highlighter. Every variant is highlighted up front, so switching language is instant.
const html = ref<Partial<Record<CodeLang, string>>>({})
onMounted(async () => {
  const highlighter = await useHighlighter()
  html.value = Object.fromEntries(
    variants.value.map(({ lang, variant }) => [lang, highlighter.codeToHtml(variant.code.trim(), { lang, theme: "red-queen-ink" })]),
  )
})

const copied = ref(false)
async function copyCode() {
  try {
    await navigator.clipboard.writeText(code.value)
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
  <div class="not-prose well overflow-hidden">
    <div class="flex items-center justify-between gap-3 border-b border-line px-2 sm:px-3">
      <div v-if="variants.length > 1" class="flex min-w-0 items-center" role="tablist" aria-label="Language">
        <button
          v-for="v in variants"
          :key="v.lang"
          role="tab"
          :aria-selected="v.lang === active.lang"
          class="-mb-px border-b-2 px-2.5 py-2 font-mono text-[11px] tracking-wide transition"
          :class="v.lang === active.lang ? 'border-queen-400 text-fg' : 'border-transparent text-fg-subtle hover:text-fg-muted'"
          @click="setPreferred(v.lang)"
        >
          {{ CODE_LANG_LABELS[v.lang] }}
        </button>
      </div>
      <span v-else class="label flex items-center gap-2 px-1.5 py-2"><span class="size-1.5 rotate-45 bg-queen-400" />{{ CODE_LANG_LABELS[active.lang] }}</span>

      <div class="flex min-w-0 items-center gap-3">
        <span
          v-if="active.variant.source"
          class="hidden min-w-0 truncate font-mono text-[10.5px] text-fg-subtle sm:block"
          :title="`The project's own code: ${active.variant.source}`"
        >{{ active.variant.source }}</span>
        <span
          v-else-if="hasRealSource && active.lang !== 'pseudo'"
          class="hidden font-mono text-[10.5px] text-fg-subtle/80 italic sm:block"
          title="Written for this page from the project's code in another language"
        >translation</span>
        <button class="shrink-0 py-2 font-mono text-[11px] text-fg-subtle transition hover:text-fg" @click="copyCode">
          {{ copied ? "Copied!" : "Copy" }}
        </button>
      </div>
    </div>
    <div v-if="html[active.lang]" class="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed [&_pre]:!bg-transparent" v-html="html[active.lang]" />
    <pre v-else class="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-fg-muted">{{ code }}</pre>
  </div>
</template>
