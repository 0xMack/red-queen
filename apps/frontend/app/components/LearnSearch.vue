<script setup lang="ts">
import { learnChapters } from "~/data/learnChapters"

// Search across every chapter's title, summary, tags, and section headings -- results link straight
// to the matching section's anchor. Press "/" anywhere on a Learn page to focus it; arrow keys +
// Enter to navigate results.
const props = withDefaults(defineProps<{ autofocusShortcut?: boolean; placeholder?: string }>(), {
  autofocusShortcut: true,
  placeholder: "Search chapters, topics, sections…",
})

interface Result {
  key: string
  chapterTitle: string
  number: number
  label: string
  kind: "chapter" | "section" | "tag"
  to: string | null
  available: boolean
}

const query = ref("")
const focused = ref(false)
const activeIndex = ref(0)
const input = ref<HTMLInputElement | null>(null)

const results = computed<Result[]>(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return []
  const terms = q.split(/\s+/)
  const matches = (text: string) => terms.every((t) => text.toLowerCase().includes(t))
  const out: Result[] = []
  learnChapters.forEach((c, i) => {
    const available = c.status === "available"
    const base = { chapterTitle: c.title, number: i + 1, available }
    if (matches(`${c.title} ${c.summary} ${c.part}`)) {
      out.push({ ...base, key: c.slug, label: c.summary, kind: "chapter", to: available ? c.path : null })
    }
    for (const s of c.sections ?? []) {
      if (matches(s)) out.push({ ...base, key: `${c.slug}#${s}`, label: s, kind: "section", to: available ? `${c.path}#${slugify(s)}` : null })
    }
    const tag = c.tags.find((t) => matches(t))
    if (tag && !out.some((r) => r.key === c.slug)) {
      out.push({ ...base, key: `${c.slug}@${tag}`, label: `tagged “${tag}”`, kind: "tag", to: available ? c.path : null })
    }
  })
  return out.slice(0, 10)
})

watch(query, () => (activeIndex.value = 0))

function go(result: Result | undefined) {
  if (!result?.to) return
  query.value = ""
  input.value?.blur()
  navigateTo(result.to)
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "ArrowDown") {
    event.preventDefault()
    activeIndex.value = Math.min(results.value.length - 1, activeIndex.value + 1)
  } else if (event.key === "ArrowUp") {
    event.preventDefault()
    activeIndex.value = Math.max(0, activeIndex.value - 1)
  } else if (event.key === "Enter") {
    go(results.value[activeIndex.value])
  } else if (event.key === "Escape") {
    query.value = ""
    input.value?.blur()
  }
}

function onGlobalKey(event: KeyboardEvent) {
  const target = event.target as HTMLElement | null
  if (event.key !== "/" || target?.closest("input, textarea, [contenteditable], [tabindex='0']")) return
  event.preventDefault()
  input.value?.focus()
}
onMounted(() => props.autofocusShortcut && window.addEventListener("keydown", onGlobalKey))
onUnmounted(() => window.removeEventListener("keydown", onGlobalKey))
</script>

<template>
  <div class="relative">
    <svg viewBox="0 0 20 20" class="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-fg-subtle" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="9" cy="9" r="6" /><path d="m14 14 4 4" stroke-linecap="round" />
    </svg>
    <input
      ref="input"
      v-model="query"
      type="search"
      :placeholder="placeholder"
      class="w-full rounded-lg border border-line bg-sunken py-2.5 pr-10 pl-9 text-sm placeholder:text-fg-subtle focus:border-queen-400/60 focus:outline-none"
      @focus="focused = true"
      @blur="focused = false"
      @keydown="onKeydown"
    >
    <kbd class="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 rounded border border-line-strong px-1.5 font-mono text-[10px] text-fg-subtle">/</kbd>

    <div
      v-if="query.trim()"
      class="absolute inset-x-0 top-full z-40 mt-2 overflow-hidden rounded-xl border border-line-strong bg-raised shadow-2xl"
    >
      <p v-if="results.length === 0" class="px-4 py-3 text-sm text-fg-subtle">No chapters or sections match “{{ query }}”.</p>
      <button
        v-for="(r, i) in results"
        :key="r.key"
        class="flex w-full items-start gap-3 px-4 py-2.5 text-left transition"
        :class="[i === activeIndex ? 'bg-bg/60' : '', r.to ? 'cursor-pointer' : 'cursor-default opacity-60']"
        @mousedown.prevent="go(r)"
        @mouseenter="activeIndex = i"
      >
        <span class="mt-0.5 font-mono text-[11px] text-fg-subtle">{{ String(r.number).padStart(2, "0") }}</span>
        <span class="min-w-0">
          <span class="block truncate text-sm font-medium text-fg">
            <span v-if="r.kind === 'section'" class="text-queen-300">§ </span>{{ r.kind === "section" ? r.label : r.chapterTitle }}
          </span>
          <span class="block truncate text-xs text-fg-subtle">
            {{ r.kind === "section" ? r.chapterTitle : r.label }}{{ r.available ? "" : " · coming soon" }}
          </span>
        </span>
      </button>
    </div>
  </div>
</template>
