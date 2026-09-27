// A chapter's "on this page" outline, built at runtime from its own <h2>s (so chapters stay plain hand-authored
// markup): gives each heading an id (slugify(text) -- the same anchors the Learn search's `sections` index links
// to), tracks which one is being read, and honors a #hash that arrived before the ids existed. Rescans whenever
// the body's DOM changes, since chapters swap in through <NuxtPage> with a transition.
export function useChapterOutline(body: Ref<HTMLElement | null>) {
  const route = useRoute()
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

    if (route.hash) document.getElementById(decodeURIComponent(route.hash.slice(1)))?.scrollIntoView()
  }

  watch(body, (el) => {
    mutationObserver?.disconnect()
    if (!el) return
    mutationObserver = new MutationObserver(() => scan())
    mutationObserver.observe(el, { childList: true, subtree: true })
    scan()
  })
  watch(
    () => route.path,
    () => {
      activeId.value = null
      nextTick(scan)
    },
  )
  onUnmounted(() => {
    mutationObserver?.disconnect()
    intersectionObserver?.disconnect()
  })

  return { outline, activeId }
}

/** How far down the page the reader is, 0..1. */
export function useReadingProgress() {
  const progress = ref(0)
  function onScroll() {
    const el = document.documentElement
    const max = el.scrollHeight - el.clientHeight
    progress.value = max > 0 ? Math.min(1, el.scrollTop / max) : 0
  }
  onMounted(() => window.addEventListener("scroll", onScroll, { passive: true }))
  onUnmounted(() => window.removeEventListener("scroll", onScroll))
  return progress
}
