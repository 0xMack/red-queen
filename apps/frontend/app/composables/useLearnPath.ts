import { chapterBySlug, resolvePath, type LearnChapter, type LearnPath } from "~/data/learnChapters"

export interface PathPosition {
  path: LearnPath
  index: number // 0-based stop on the path
  stops: LearnChapter[]
  prev: LearnChapter | null
  next: LearnChapter | null
}

/** Where a chapter sits on the path it's being read on (`?path=`, else its home path) -- what the sidebar, the title
 *  page and prev/next all follow, so switching paths mid-chapter changes all three together. */
export function useLearnPath(chapter: Ref<LearnChapter>): ComputedRef<PathPosition> {
  const route = useRoute()
  return computed(() => {
    const requested = typeof route.query.path === "string" ? route.query.path : null
    const path = resolvePath(chapter.value, requested)
    const stops = path.chapters.map((slug) => chapterBySlug(slug)!)
    const index = path.chapters.indexOf(chapter.value.slug)
    return { path, index, stops, prev: stops[index - 1] ?? null, next: stops[index + 1] ?? null }
  })
}
