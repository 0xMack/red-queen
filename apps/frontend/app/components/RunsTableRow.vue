<script setup lang="ts">
import type { RunRow } from "~/utils/runMeta"

// One run in the runs table (pages/runs/index.vue): its own row, or one of an experiment group's rows (`nested`).
defineProps<{ r: RunRow; now: number; nested?: boolean }>()
</script>

<template>
  <tr
    class="group cursor-pointer border-b border-line/60 transition last:border-0 hover:bg-raised/60"
    @click="navigateTo(`/runs/${r.run.run_id}`)"
  >
    <td class="py-3 pr-4" :class="nested ? 'pl-10' : 'pl-4'">
      <NuxtLink :to="`/runs/${r.run.run_id}`" class="font-medium text-fg group-hover:text-queen-300" @click.stop>
        {{ r.label }}
      </NuxtLink>
      <p class="mt-0.5 flex items-center gap-2 font-mono text-[11px] text-fg-subtle">
        {{ shortId(r.run.run_id) }}
        <span v-if="r.meta.note" class="truncate font-sans italic">· {{ r.meta.note }}</span>
      </p>
    </td>
    <td class="px-3 py-3"><StatusBadge :status="r.run.status" :stale="r.stale" /></td>
    <td class="px-3 py-3">
      <div class="flex flex-col gap-1">
        <span v-if="r.meta.selection" class="chip w-fit">{{ r.meta.selection }}</span>
        <span
          v-if="r.meta.seedStrategy"
          class="chip w-fit"
          :class="r.meta.seedStrategy.startsWith('resample') ? 'text-life-300' : ''"
          title="Training seeds: fixed = the same games every generation; resample = fresh games every generation"
        >
          seeds {{ r.meta.seedStrategy }}
        </span>
        <span v-if="r.meta.variation" class="w-fit max-w-40 truncate font-mono text-[11px] text-fg-subtle" :title="r.meta.variation">{{ r.meta.variation }}</span>
        <span v-if="!r.meta.selection && !r.meta.variation" class="text-fg-subtle">--</span>
      </div>
    </td>
    <td class="num px-3 py-3 text-right text-fg-muted">{{ r.meta.populationSize ?? "--" }}</td>
    <td class="px-3 py-3 font-mono text-xs whitespace-nowrap text-fg-muted">{{ r.genome }}</td>
    <td class="px-3 py-3">
      <div class="flex items-center gap-2">
        <span class="num w-16 text-xs text-fg-muted">
          {{ r.generations }}<span v-if="r.meta.targetGenerations" class="text-fg-subtle">/{{ r.meta.targetGenerations }}</span>
        </span>
        <div v-if="r.meta.targetGenerations" class="h-1.5 w-20 overflow-hidden rounded-full bg-raised">
          <div
            class="h-full rounded-full"
            :class="r.run.status === 'running' && !r.stale ? 'bg-life-400' : 'bg-fg-subtle'"
            :style="{ width: `${Math.min(100, (r.generations / r.meta.targetGenerations) * 100)}%` }"
          />
        </div>
      </div>
    </td>
    <td class="num px-3 py-3 text-right text-fg-muted">{{ formatFitness(r.best, 3) }}</td>
    <td class="px-3 py-3 text-right whitespace-nowrap" data-held-out>
      <NuxtLink
        v-if="r.heldOut"
        :to="`/games/snake?watch=${encodeURIComponent(r.heldOut.entrantId)}`"
        class="group/lb inline-flex flex-col items-end"
        title="Open on the leaderboard"
        @click.stop
      >
        <span class="num font-semibold text-fg group-hover/lb:text-queen-300">{{ r.heldOut.mean.toFixed(2) }}</span>
        <span class="text-[10px] text-fg-subtle">🏆 #{{ r.heldOut.rank }} of {{ r.heldOut.of }}</span>
      </NuxtLink>
      <span v-else-if="unrankedReason(r.run)" class="cursor-help text-fg-subtle" :title="unrankedReason(r.run)!">--</span>
      <span v-else class="text-fg-subtle">--</span>
    </td>
    <td class="px-3 py-3"><Sparkline :values="r.trend" class="h-8 w-24" /></td>
    <td class="px-3 py-3 whitespace-nowrap text-fg-muted" :title="formatTimestamp(r.run.created_at)">
      {{ formatRelative(r.run.created_at, now) }}
    </td>
    <td class="num px-3 py-3 text-right text-fg-muted">{{ formatDuration(r.duration) }}</td>
    <td class="px-4 py-3 text-right">
      <NuxtLink v-if="r.meta.watchable" :to="`/runs/${r.run.run_id}`" class="btn-ghost btn-sm whitespace-nowrap" @click.stop>
        ▶ Watch
      </NuxtLink>
    </td>
  </tr>
</template>
