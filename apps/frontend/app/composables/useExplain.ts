import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"
import type { ExplainRef, ResolvedExplainer } from "~/types/explain"
import { EMPTY_CONTEXT, formatRef, parseRef, resolveExplainer, type ExplainContext } from "~/data/explainers"

// Explainers' shared state. The side panel lives once in app.vue and is opened by the URL (`?explain=kind:id`), so a
// view can be linked; ⓘ popovers are wherever a name is. Both resolve refs against the current page's leaderboard
// (`ExplainContext`), which a game page registers while it's mounted -- module-level (the panel isn't below the page
// in the component tree, so provide/inject can't reach it), set only on the client, and held shallowly: records are
// read, never mutated.

const context = shallowRef<ExplainContext>(EMPTY_CONTEXT)
/** The one open popover (opening another closes it). */
const openTip = ref<string | null>(null)

/** Registers this page's leaderboard for explainers while the calling component is mounted. */
export function useExplainContext(source: () => { game: string; entries: EvaluationRecord[]; interfacesById: Record<string, InterfaceInfo> }) {
  onMounted(() => {
    const stop = watch(
      source,
      (s) => {
        context.value = markRaw({ game: s.game, entries: s.entries, interfacesById: s.interfacesById })
      },
      { immediate: true },
    )
    onUnmounted(() => {
      stop()
      context.value = EMPTY_CONTEXT
    })
  })
}

export function useExplain() {
  const route = useRoute()
  const router = useRouter()

  const current = computed<ExplainRef | null>(() => parseRef(route.query.explain as string | undefined))

  function open(ref: ExplainRef | string) {
    openTip.value = null
    const value = typeof ref === "string" ? ref : formatRef(ref)
    // push, not replace: Back closes the panel (and walks back through explainers opened from each other).
    router.push({ query: { ...route.query, explain: value } })
  }
  function close() {
    const { explain: _, ...rest } = route.query
    router.replace({ query: rest })
  }
  const resolve = (ref: ExplainRef | string): ResolvedExplainer | null => {
    const parsed = typeof ref === "string" ? parseRef(ref) : ref
    return parsed ? resolveExplainer(parsed, context.value) : null
  }

  return { current, open, close, resolve, context: computed(() => context.value), openTip }
}
