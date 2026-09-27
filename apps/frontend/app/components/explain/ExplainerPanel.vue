<script setup lang="ts">
// The full explainer: a side panel over the right of the page, opened by `?explain=kind:id` (so a view can be linked,
// and Back closes it), mounted once in app.vue. Same resolved explainer as the ⓘ card, all of it: the picture and a
// live demo, this entrant's own settings and provenance, how it works, the rule as code, where it shines and fails,
// results, and the way on -- chapters, the training run, related explainers (which open here, in place).
const { current, close, open, resolve, context } = useExplain()

// Re-resolved when the page's records arrive too (a linked panel opens before the leaderboard loads).
const explainer = computed(() => {
  void context.value
  return current.value ? resolve(current.value) : null
})
const visible = computed(() => !!explainer.value)

const body = ref<HTMLElement | null>(null)
watch(
  () => current.value && `${current.value.kind}:${current.value.id}`,
  () => nextTick(() => body.value?.scrollTo({ top: 0 })),
)

function onKey(e: KeyboardEvent) {
  if (e.key === "Escape" && visible.value) close()
}
onMounted(() => window.addEventListener("keydown", onKey))
onUnmounted(() => window.removeEventListener("keydown", onKey))

const KIND_LABEL = { algorithm: "Algorithm", entrant: "Entrant", scenario: "Scenario", representation: "Representation", metric: "Measure" } as const
</script>

<template>
  <Transition name="fade">
    <div v-if="visible" class="fixed inset-0 z-[60] bg-black/50 backdrop-blur-[1px] xl:bg-black/25" aria-hidden="true" @click="close" />
  </Transition>
  <Transition name="slide">
    <aside
      v-if="visible && explainer"
      class="panel fixed inset-y-0 right-0 z-[61] flex w-full flex-col border-l border-line-strong sm:w-[min(38rem,92vw)]"
      role="dialog"
      aria-modal="true"
      :aria-label="explainer.title"
    >
      <header class="ticks flex shrink-0 items-start gap-4 border-b border-line px-5 pt-5 pb-4 sm:px-7">
        <div class="min-w-0 flex-1">
          <p class="label flex flex-wrap items-center gap-x-2">
            <span class="text-queen-300">{{ KIND_LABEL[explainer.kind] }}</span>
            <span class="normal-case tracking-normal">{{ explainer.eyebrow }}</span>
          </p>
          <h2 class="mt-2 text-[2.1rem] leading-[1.05]">{{ explainer.title }}</h2>
        </div>
        <button type="button" class="btn-quiet btn-sm -mr-2 text-base" aria-label="Close" @click="close">✕</button>
      </header>

      <div ref="body" class="min-h-0 flex-1 space-y-8 overflow-y-auto px-5 py-6 sm:px-7">
        <p class="text-[15px] leading-relaxed text-fg-muted">{{ explainer.summary }}</p>

        <ExplainVisual v-if="explainer.visual" :spec="explainer.visual" />

        <section v-if="explainer.instance">
          <h3 class="label mb-3 text-fg-muted">{{ explainer.instance.title }}</h3>
          <p v-if="explainer.instance.subtitle" class="mb-3 text-sm leading-relaxed text-fg-muted">{{ explainer.instance.subtitle }}</p>
          <ExplainFacts :facts="explainer.instance.facts" meanings />
          <div v-if="explainer.instance.chips?.length" class="mt-3 flex flex-wrap gap-1">
            <span v-for="(c, i) in explainer.instance.chips" :key="i" class="chip"><span class="text-fg-subtle">{{ i }}</span> {{ c }}</span>
          </div>
          <p v-for="n in explainer.instance.notes" :key="n" class="mt-3 text-xs leading-relaxed text-fg-subtle">{{ n }}</p>
        </section>

        <section v-if="explainer.how?.length">
          <h3 class="label mb-3 text-fg-muted">How it works</h3>
          <ol class="steps space-y-2.5 text-sm leading-relaxed text-fg-muted">
            <li v-for="(s, i) in explainer.how" :key="i" class="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-2">
              <span class="num pt-px text-xs text-queen-300">{{ String(i + 1).padStart(2, "0") }}</span>
              <span>{{ s }}</span>
            </li>
          </ol>
          <CodeBlock v-if="explainer.code" class="mt-4" lang="python" :code="explainer.code" />
        </section>

        <section v-if="explainer.live">
          <h3 class="label mb-3 text-fg-muted">{{ explainer.live.kind === "observer" ? "Watch it look" : "Watch it play" }}</h3>
          <ExplainLive :key="`${explainer.ref.kind}:${explainer.ref.id}`" :spec="explainer.live" />
        </section>

        <section v-if="explainer.good?.length || explainer.bad?.length" class="grid gap-4 sm:grid-cols-2">
          <div v-if="explainer.good?.length" class="well p-3.5">
            <h3 class="label mb-2 text-life-300">Where it shines</h3>
            <p v-for="g in explainer.good" :key="g" class="text-[13px] leading-relaxed text-fg-muted">{{ g }}</p>
          </div>
          <div v-if="explainer.bad?.length" class="well p-3.5">
            <h3 class="label mb-2 text-queen-300">Where it struggles</h3>
            <p v-for="b in explainer.bad" :key="b" class="text-[13px] leading-relaxed text-fg-muted">{{ b }}</p>
          </div>
        </section>

        <section v-if="explainer.results?.rows.length">
          <h3 class="label mb-3 text-fg-muted">{{ explainer.results.title }}</h3>
          <ExplainResults :results="explainer.results" @open="open" />
        </section>

        <section v-if="explainer.links.length">
          <h3 class="label mb-3 text-fg-muted">Read more</h3>
          <ExplainLinks :links="explainer.links" />
        </section>

        <section v-if="explainer.related.length" class="pb-4">
          <h3 class="label mb-3 text-fg-muted">Related</h3>
          <div class="flex flex-wrap gap-1.5">
            <button v-for="r in explainer.related" :key="r.ref" type="button" class="chip transition hover:border-queen-300 hover:text-fg" @click="open(r.ref)">
              <span class="text-queen-300">i</span> {{ r.label }}
            </button>
          </div>
        </section>
      </div>
    </aside>
  </Transition>
</template>

<style scoped>
.panel {
  background:
    radial-gradient(90% 40% at 100% 0%, color-mix(in srgb, var(--color-queen-500) 7%, transparent), transparent 70%),
    var(--color-bg);
  box-shadow: -24px 0 60px rgb(0 0 0 / 0.55);
}
.slide-enter-active,
.slide-leave-active {
  transition: transform 0.24s cubic-bezier(0.22, 1, 0.36, 1);
}
.slide-enter-from,
.slide-leave-to {
  transform: translateX(100%);
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
