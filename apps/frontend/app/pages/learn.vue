<script setup lang="ts">
import { chapterNumber, learnChapters } from "~/data/learnChapters"

// Parent route for every /learn/* page. The index (/learn) renders full-width on its own; a chapter
// gets the "textbook" frame: chapter nav + search on the left, a generated header/cover and
// prev/next links around the chapter body, and an "on this page" outline on the right built from
// the chapter's own <h2>s at runtime (so chapters stay plain hand-authored markup).
const route = useRoute()

const chapter = computed(() => learnChapters.find((c) => c.path === route.path.replace(/\/$/, "")) ?? null)
const isIndex = computed(() => !chapter.value)
const index = computed(() => (chapter.value ? learnChapters.indexOf(chapter.value) : -1))
const available = learnChapters.filter((c) => c.status === "available")
const prev = computed(() => {
  const i = available.findIndex((c) => c.slug === chapter.value?.slug)
  return i > 0 ? available[i - 1] : null
})
const next = computed(() => {
  const i = available.findIndex((c) => c.slug === chapter.value?.slug)
  return i >= 0 && i < available.length - 1 ? available[i + 1] : null
})
const prerequisites = computed(() =>
  (chapter.value?.prerequisites ?? []).map((slug) => learnChapters.find((c) => c.slug === slug)).filter((c) => !!c),
)

useHead({ title: () => chapter.value?.title ?? "Learn" })

const parts = computed(() => {
  const groups: { part: string; chapters: { chapter: (typeof learnChapters)[number]; number: number }[] }[] = []
  learnChapters.forEach((c, i) => {
    let group = groups.find((g) => g.part === c.part)
    if (!group) groups.push((group = { part: c.part, chapters: [] }))
    group.chapters.push({ chapter: c, number: i + 1 })
  })
  return groups
})

// "On this page" -------------------------------------------------------------------------------------
const body = ref<HTMLElement | null>(null)
const outline = ref<{ id: string; text: string }[]>([])
const activeId = ref<string | null>(null)
let mutationObserver: MutationObserver | null = null
let intersectionObserver: IntersectionObserver | null = null

function scan() {
  const headings = [...(body.value?.querySelectorAll<HTMLElement>(".prose-chapter h2") ?? [])]
  for (const h of headings) if (!h.id) h.id = slugify(h.textContent ?? "")
  const next = headings.map((h) => ({ id: h.id, text: h.textContent?.trim() ?? "" }))
  if (JSON.stringify(next) === JSON.stringify(outline.value)) return
  outline.value = next

  intersectionObserver?.disconnect()
  intersectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
      if (visible[0]) activeId.value = visible[0].target.id
    },
    { rootMargin: "-80px 0px -65% 0px" },
  )
  headings.forEach((h) => intersectionObserver!.observe(h))

  // A hash link from search lands before the ids exist on first render -- honor it once they do.
  if (route.hash) document.getElementById(decodeURIComponent(route.hash.slice(1)))?.scrollIntoView()
}

watch(body, (el) => {
  mutationObserver?.disconnect()
  if (!el) return
  // Chapters swap in via <NuxtPage> (with a transition), so rescan whenever the body's DOM changes.
  mutationObserver = new MutationObserver(() => scan())
  mutationObserver.observe(el, { childList: true, subtree: true })
  scan()
})
watch(() => route.path, () => {
  activeId.value = null
  nextTick(scan)
})
onUnmounted(() => {
  mutationObserver?.disconnect()
  intersectionObserver?.disconnect()
})

const mobileNavOpen = ref(false)
watch(() => route.fullPath, () => (mobileNavOpen.value = false))

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" })
}

// Reading progress bar.
const progress = ref(0)
function onScroll() {
  const el = document.documentElement
  const max = el.scrollHeight - el.clientHeight
  progress.value = max > 0 ? Math.min(1, el.scrollTop / max) : 0
}
onMounted(() => window.addEventListener("scroll", onScroll, { passive: true }))
onUnmounted(() => window.removeEventListener("scroll", onScroll))
</script>

<template>
  <div>
    <NuxtPage v-if="isIndex" />

    <div v-else class="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8">
      <div class="fixed inset-x-0 top-14 z-40 h-0.5 bg-transparent">
        <div class="h-full bg-queen-400 shadow-[0_0_10px_rgb(255_92_122/0.8)] transition-[width] duration-150" :style="{ width: `${progress * 100}%` }" />
      </div>

      <div class="grid gap-8 py-8 lg:grid-cols-[260px_minmax(0,1fr)] xl:grid-cols-[260px_minmax(0,1fr)_220px]">
        <!-- Chapter nav -->
        <aside class="lg:sticky lg:top-20 lg:h-[calc(100vh-6rem)] lg:overflow-y-auto lg:pb-8">
          <LearnSearch />
          <button
            class="btn-ghost mt-3 w-full justify-between lg:hidden"
            :aria-expanded="mobileNavOpen"
            @click="mobileNavOpen = !mobileNavOpen"
          >
            <span>Chapter {{ index + 1 }} of {{ learnChapters.length }}</span>
            <span>{{ mobileNavOpen ? "▴" : "▾" }}</span>
          </button>
          <nav class="mt-5 space-y-6" :class="mobileNavOpen ? 'block' : 'hidden lg:block'">
            <NuxtLink to="/learn" class="flex items-center gap-2 text-sm text-fg-subtle transition hover:text-fg">
              <span>←</span> All chapters
            </NuxtLink>
            <div v-for="group in parts" :key="group.part">
              <p class="mb-2 font-mono text-[10px] tracking-[0.18em] text-fg-subtle uppercase">{{ group.part }}</p>
              <ul class="space-y-0.5 border-l border-line">
                <li v-for="{ chapter: c, number } in group.chapters" :key="c.slug">
                  <NuxtLink
                    v-if="c.status === 'available'"
                    :to="c.path"
                    class="-ml-px flex gap-2.5 border-l py-1.5 pr-2 pl-3 text-sm transition"
                    :class="c.slug === chapter?.slug ? 'border-queen-400 font-medium text-fg' : 'border-transparent text-fg-muted hover:border-fg-subtle hover:text-fg'"
                  >
                    <span class="num text-[11px] leading-5 text-fg-subtle">{{ String(number).padStart(2, "0") }}</span>
                    <span>{{ c.title }}</span>
                  </NuxtLink>
                  <span v-else class="-ml-px flex gap-2.5 border-l border-transparent py-1.5 pr-2 pl-3 text-sm text-fg-subtle/70">
                    <span class="num text-[11px] leading-5">{{ String(number).padStart(2, "0") }}</span>
                    <span>{{ c.title }} <span class="text-[10px] tracking-wide uppercase">· soon</span></span>
                  </span>
                </li>
              </ul>
            </div>
          </nav>
        </aside>

        <!-- Chapter -->
        <div class="min-w-0">
          <header v-if="chapter" class="card relative overflow-hidden">
            <div class="grid md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <div class="relative z-10 p-6 sm:p-8">
                <p class="eyebrow">Chapter {{ chapterNumber(chapter.slug) }} · {{ chapter.part }}</p>
                <h1 class="mt-3 text-3xl leading-tight font-semibold sm:text-4xl">{{ chapter.title }}</h1>
                <p class="mt-3 text-fg-muted">{{ chapter.summary }}</p>
                <div class="mt-5 flex flex-wrap items-center gap-2 text-xs text-fg-subtle">
                  <span v-if="chapter.readMinutes" class="chip text-fg">{{ chapter.readMinutes }} min read</span>
                  <span v-for="tag in chapter.tags" :key="tag" class="chip">{{ tag }}</span>
                </div>
                <p v-if="prerequisites.length" class="mt-4 text-sm text-fg-subtle">
                  Builds on
                  <template v-for="(p, i) in prerequisites" :key="p.slug">
                    <NuxtLink :to="p.path" class="link">{{ p.title }}</NuxtLink><span v-if="i < prerequisites.length - 1">, </span>
                  </template>.
                </p>
              </div>
              <div class="relative hidden min-h-56 border-l border-line md:block">
                <img v-if="chapter.image" :src="chapter.image" :alt="`${chapter.title} cover`" class="absolute inset-0 size-full object-cover object-top">
                <ChapterArt v-else :kind="chapter.art" fit="contain" class="absolute inset-0" />
                <div class="absolute inset-0 bg-gradient-to-r from-surface via-surface/10 to-transparent" />
              </div>
            </div>
          </header>

          <div ref="body" class="mx-auto mt-10 max-w-3xl">
            <NuxtPage />
          </div>

          <nav class="mx-auto mt-16 grid max-w-3xl gap-3 sm:grid-cols-2">
            <NuxtLink v-if="prev" :to="prev.path" class="card card-hover block p-4">
              <p class="text-xs text-fg-subtle">← Previous</p>
              <p class="mt-1 font-medium">{{ prev.title }}</p>
            </NuxtLink>
            <span v-else />
            <NuxtLink v-if="next" :to="next.path" class="card card-hover block p-4 text-right">
              <p class="text-xs text-fg-subtle">Next →</p>
              <p class="mt-1 font-medium">{{ next.title }}</p>
            </NuxtLink>
          </nav>
        </div>

        <!-- On this page -->
        <aside class="hidden xl:block">
          <div v-if="outline.length" class="sticky top-20">
            <p class="font-mono text-[10px] tracking-[0.18em] text-fg-subtle uppercase">On this page</p>
            <ul class="mt-3 space-y-1 border-l border-line text-sm">
              <li v-for="h in outline" :key="h.id">
                <a
                  :href="`#${h.id}`"
                  class="-ml-px block border-l py-1 pl-3 transition"
                  :class="activeId === h.id ? 'border-queen-400 text-fg' : 'border-transparent text-fg-subtle hover:text-fg'"
                >
                  {{ h.text }}
                </a>
              </li>
            </ul>
            <button class="mt-6 text-xs text-fg-subtle transition hover:text-fg" @click="scrollToTop">
              ↑ Back to top
            </button>
          </div>
        </aside>
      </div>
    </div>
  </div>
</template>
