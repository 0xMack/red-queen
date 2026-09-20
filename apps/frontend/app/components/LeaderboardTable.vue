<script setup lang="ts">
import type { LeaderboardColumn, ScoreSpec } from "~/games/types"
import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"

// The full leaderboard table (docs/design/0007): quality, inference, and training measurements per
// entrant, in rank order. Clicking a row selects that entrant (the game page then plays it); the run
// id links out to the run itself. Game-agnostic: the score's name and format come from the game's
// ScoreSpec, and a game adds its own columns (Snake: score spread and train gap; Checkers: its
// win/draw/loss record) between the score and the cost columns.
const props = defineProps<{
  entries: EvaluationRecord[]
  interfacesById: Record<string, InterfaceInfo>
  score: ScoreSpec
  columns?: LeaderboardColumn[]
  selectedId?: string | null
}>()
defineEmits<{ select: [entrantId: string] }>()

// Rank + statistical ties: overlapping 95% intervals with the entry above = not a real difference.
const ranked = computed(() =>
  props.entries.map((r, i) => {
    const prev = props.entries[i - 1]
    const q = r.metrics.quality
    const tiedWithPrev = !!prev && Math.abs(prev.metrics.quality.mean - q.mean) <= prev.metrics.quality.ci95 + q.ci95
    return { record: r, rank: i + 1, tiedWithPrev }
  }),
)
const maxMean = computed(() => Math.max(props.score.scaleMin, ...props.entries.map((r) => r.metrics.quality.mean + r.metrics.quality.ci95)))
</script>

<template>
  <section class="card overflow-x-auto">
    <table class="w-full text-sm">
      <thead>
        <tr class="border-b border-line text-left text-[11px] tracking-wide text-fg-subtle uppercase">
          <th class="px-4 py-3 font-medium">#</th>
          <th class="px-3 py-3 font-medium">Entrant</th>
          <th class="px-3 py-3 font-medium">Representation</th>
          <th class="px-3 py-3 font-medium">{{ score.label }}</th>
          <th v-for="c in columns" :key="c.id" class="px-3 py-3 text-right font-medium" :title="c.title">{{ c.header }}</th>
          <th class="px-3 py-3 text-right font-medium" title="per decision: encoding the board + running the model">Inference</th>
          <th class="px-3 py-3 text-right font-medium">Params</th>
          <th class="px-3 py-3 text-right font-medium">Training</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="{ record: r, rank, tiedWithPrev } in ranked"
          :id="`row-${r.entrant_id}`"
          :key="r.entrant_id"
          class="cursor-pointer border-b border-line/50 transition last:border-0"
          :class="selectedId === r.entrant_id ? 'bg-queen-500/10' : 'hover:bg-raised/50'"
          @click="$emit('select', r.entrant_id)"
        >
          <td class="num px-4 py-3 text-fg-muted">
            {{ rank }}
            <span v-if="tiedWithPrev" class="ml-1 text-[10px] text-fg-subtle" title="95% intervals overlap with the entrant above: not a statistically clear difference">≈</span>
          </td>
          <td class="px-3 py-3">
            <div class="flex items-center gap-2">
              <span class="size-2 shrink-0 rounded-full" :style="{ background: entrantColor(r) }" />
              <span class="font-medium" :title="r.label">{{ entrantShortLabel(r) }}</span>
              <span v-if="selectedId === r.entrant_id" class="rounded-full bg-queen-500/20 px-2 py-0.5 text-[10px] font-medium text-queen-200">watching</span>
            </div>
            <p class="mt-0.5 pl-4 text-[11px] text-fg-subtle">
              <span class="capitalize">{{ r.entrant_kind }}</span>
              <template v-if="r.entrant_kind === 'champion'"> · {{ entrantDetail(r).split(" · ").slice(0, -1).join(" · ") }}</template>
              <NuxtLink v-if="r.run_id" :to="`/runs/${r.run_id}`" class="font-mono hover:text-queen-300" @click.stop> · {{ shortId(r.run_id) }} ↗</NuxtLink>
              <span v-if="r.metrics.model.note" class="italic"> · {{ r.metrics.model.note }}</span>
            </p>
          </td>
          <td class="px-3 py-3">
            <span class="chip" :style="{ color: LEVEL_COLORS[r.metrics.model.observer_level] }">L{{ r.metrics.model.observer_level }} {{ interfacesById[r.interface]?.observer.level_name ?? "" }}</span>
            <p class="mt-1 font-mono text-[10px] text-fg-subtle">{{ r.interface }}</p>
          </td>
          <td class="min-w-48 px-3 py-3">
            <div class="flex items-baseline gap-2">
              <span class="num text-base font-semibold">{{ score.format(r.metrics.quality.mean) }}</span>
              <span class="num text-[11px] text-fg-subtle">± {{ score.format(r.metrics.quality.ci95) }}</span>
            </div>
            <div class="relative mt-1.5 h-1.5 rounded-full bg-raised">
              <div class="absolute inset-y-0 left-0 rounded-full" :style="{ width: `${(r.metrics.quality.mean / maxMean) * 100}%`, background: entrantColor(r) }" />
              <div
                class="absolute -inset-y-0.5 rounded-full bg-fg/25"
                :style="{
                  left: `${(Math.max(0, r.metrics.quality.mean - r.metrics.quality.ci95) / maxMean) * 100}%`,
                  width: `${((2 * r.metrics.quality.ci95) / maxMean) * 100}%`,
                }"
              />
            </div>
          </td>
          <td v-for="c in columns" :key="c.id" class="num px-3 py-3 text-right text-xs">
            <template v-if="c.cell(r)">
              <span :class="c.cell(r)!.tone === 'warn' ? 'text-queen-300' : c.cell(r)!.tone === 'muted' ? 'text-fg-muted' : 'text-fg'">{{ c.cell(r)!.text }}</span>
              <p v-if="c.cell(r)!.sub" class="text-[10px] text-fg-subtle">{{ c.cell(r)!.sub }}</p>
            </template>
          </td>
          <td class="num px-3 py-3 text-right text-xs">
            <span class="text-fg">{{ r.metrics.inference.total_us.toFixed(1) }} µs</span>
            <p class="text-[10px] text-fg-subtle">{{ r.metrics.inference.encode_us.toFixed(1) }} + {{ r.metrics.inference.decide_us.toFixed(1) }}</p>
          </td>
          <td class="num px-3 py-3 text-right text-xs text-fg-muted">
            {{ r.metrics.inference.parameters ? r.metrics.inference.parameters.toLocaleString() : "0" }}
            <p v-if="r.metrics.inference.artifact_bytes" class="text-[10px] text-fg-subtle">{{ compactNumber(r.metrics.inference.artifact_bytes) }}B</p>
          </td>
          <td class="num px-3 py-3 text-right text-xs">
            <span v-if="r.metrics.training.none" class="text-fg-subtle">none</span>
            <template v-else>
              <span class="text-fg" :title="r.metrics.training.measured ? 'measured' : 'estimated from config + timestamps (recorded before cost tracking)'">
                <template v-if="r.metrics.training.episodes != null">
                  {{ r.metrics.training.measured ? "" : "~" }}{{ compactNumber(r.metrics.training.episodes) }} ep
                </template>
                <template v-else-if="r.metrics.training.fitness_evaluations != null">
                  {{ compactNumber(r.metrics.training.fitness_evaluations) }} evals
                </template>
                <template v-else>? ep</template>
              </span>
              <p class="text-[10px] text-fg-subtle">
                {{ r.metrics.training.active_s != null ? formatDuration(r.metrics.training.active_s) : "--" }}
                <template v-if="r.metrics.training.cpu_s != null"> · {{ formatDuration(r.metrics.training.cpu_s) }} CPU</template>
                <template v-if="!r.metrics.training.measured"> · est.</template>
              </p>
            </template>
          </td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
