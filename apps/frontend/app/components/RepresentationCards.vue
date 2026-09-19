<script setup lang="ts">
import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"

// One card per representation (interface = observer + action adapter, docs/design/0007), with how
// many leaderboard entrants play under it and the best of them.
const props = defineProps<{ interfaces: InterfaceInfo[]; entries: EvaluationRecord[] }>()

const entrantsPerInterface = computed(() => {
  const counts: Record<string, { n: number; best: number }> = {}
  for (const r of props.entries) {
    const c = (counts[r.interface] ??= { n: 0, best: -Infinity })
    c.n += 1
    c.best = Math.max(c.best, r.metrics.quality.mean)
  }
  return counts
})
</script>

<template>
  <div class="grid gap-5 md:grid-cols-2">
    <article v-for="i in interfaces" :key="i.id" class="card p-5">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <span class="chip" :style="{ color: LEVEL_COLORS[i.observer.level] }">L{{ i.observer.level }} · {{ i.observer.level_name }}</span>
        <span class="font-mono text-[11px] text-fg-subtle">{{ i.id }}</span>
      </div>
      <h3 class="mt-3 text-lg font-semibold">{{ i.observer.id }} <span class="text-fg-subtle">+ {{ i.action.id }}</span></h3>
      <p class="mt-2 text-sm text-fg-muted">{{ i.observer.description }}</p>
      <p class="mt-2 text-sm text-fg-muted">{{ i.action.description }}</p>
      <div class="mt-4 flex flex-wrap gap-1.5">
        <template v-if="i.observer.size <= 24">
          <span v-for="(name, n) in i.observer.feature_names" :key="n" class="chip"><span class="text-fg-subtle">{{ n }}</span> {{ name }}</span>
        </template>
        <span v-else class="chip">{{ i.observer.size }} inputs: {{ i.observer.feature_names[0] }} … {{ i.observer.feature_names.at(-1) }}</span>
      </div>
      <dl class="mt-4 grid grid-cols-3 gap-2 border-t border-line pt-4 text-xs">
        <div><dt class="text-fg-subtle">inputs</dt><dd class="num text-fg">{{ i.observer.size }}</dd></div>
        <div><dt class="text-fg-subtle">outputs</dt><dd class="num text-fg">{{ i.action.num_outputs }}</dd></div>
        <div>
          <dt class="text-fg-subtle">entrants · best</dt>
          <dd class="num text-fg">
            {{ entrantsPerInterface[i.id]?.n ?? 0 }}
            <span v-if="entrantsPerInterface[i.id]" class="text-fg-subtle">· {{ entrantsPerInterface[i.id]!.best.toFixed(2) }}</span>
          </dd>
        </div>
      </dl>
    </article>
  </div>
</template>
