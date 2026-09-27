<script setup lang="ts">
import type { Reveal, RevealArm } from "~/utils/bandit"

// The end of a game (docs/design/0011): how every machine really paid -- under each lamp, and before and after a
// drift -- next to where each player's pulls went. The reveal is the lesson: the machine you gave up on after two
// losses, the jackpot nobody saw pay.
const props = defineProps<{
  reveal: Reveal
  /** Each player's pulls per machine. */
  players: { label: string; counts: number[]; you?: boolean }[]
}>()

const lamps = computed(() => props.reveal.lamps)
const best = (arms: RevealArm[]) => arms.reduce((b, a, i) => (a.mean > arms[b]!.mean ? i : b), 0)
</script>

<template>
  <div class="overflow-x-auto">
    <table class="w-full text-xs">
      <thead>
        <tr class="label border-b border-line text-left">
          <th class="py-2 pr-3 font-normal">machine</th>
          <th v-for="(_, l) in lamps" :key="l" class="py-2 pr-3 font-normal">{{ lamps.length > 1 ? (l === 0 ? "under the red lamp" : "under the blue lamp") : "how it really pays" }}</th>
          <th v-if="reveal.drift" class="py-2 pr-3 font-normal">after pull {{ reveal.drift.at }}</th>
          <th v-for="p in players" :key="p.label" class="py-2 pr-2 text-right font-normal" :class="p.you ? 'text-fg' : ''">{{ p.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(arm, a) in lamps[0]" :key="a" class="border-b border-line/50">
          <td class="py-2 pr-3 font-display text-lg leading-none" :style="{ color: armColor(a) }">{{ armName(a) }}</td>
          <td v-for="(arms, l) in lamps" :key="l" class="py-2 pr-3 text-fg-muted">
            <span :class="best(arms) === a ? 'text-gold-300' : ''">{{ describeArm(arms[a]!) }}</span>
            <span v-if="best(arms) === a" class="ml-1 text-gold-300">★</span>
          </td>
          <td v-if="reveal.drift" class="py-2 pr-3 text-fg-muted">
            <span v-if="reveal.drift.after[a] !== arm.mean" class="text-queen-300">now {{ Math.round(reveal.drift.after[a]! * 100) }}%</span>
            <span v-else class="text-fg-subtle">unchanged</span>
          </td>
          <td v-for="p in players" :key="p.label" class="num py-2 pr-2 text-right" :class="p.you ? 'text-fg' : 'text-fg-muted'">{{ p.counts[a] ?? 0 }}×</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
