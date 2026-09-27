<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { Racer } from "~/components/bandit/BanditRace.vue"

// The bandit's Play stage (docs/design/0011): you pull the machines (click them, or press 1-9) against strategies
// playing the *same* game -- same machines, and the n-th pull of a machine pays the same for everyone -- one of their
// pulls per pull of yours. Your running averages stay hidden (keeping them in your head is the game), and so does
// everyone's skill until the end, when the reveal shows how each machine really paid and where everyone's pulls went.
// On the default scenario your skill joins the side leaderboard (this browser only).
// Props spelled out (StageProps, games/types.ts): the SFC compiler can't resolve an imported type as the whole props.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()
const emit = defineEmits<{ score: [score: number, live: boolean, label?: string]; exit: [] }>()

const route = useRoute()
const scenario = computed(() => scenarioById(route.query.scenario as string | undefined))

// You, the entrant on stage, and two reference points -- greedy (the trap) and Thompson sampling -- without repeats.
const you = useBanditRun()
const rivalSlots = [useBanditRun(), useBanditRun(), useBanditRun()]
const rivals = computed(() => {
  const picked: { id: string; label: string; strategy: string; params: string }[] = []
  const add = (id: string, label: string, s: { strategy: string; params: string } | null) => {
    if (s && !picked.some((p) => p.strategy === s.strategy && p.params === s.params)) picked.push({ id, label, ...s })
  }
  if (props.entry) add(props.entry.entrant_id, props.entry.label, entrantStrategy(props.entry))
  for (const id of ["greedy", "thompson", "optimistic"]) {
    const s = BANDIT_STRATEGIES.find((x) => x.id === id)!
    const record = props.entries.find((r) => entrantStrategy(r)?.strategy === s.strategy && entrantStrategy(r)?.params === s.params)
    add(id, record?.label ?? s.label, s)
  }
  return picked.slice(0, rivalSlots.length)
})
const racers = computed<Racer[]>(() => [
  { id: "you", label: "You", run: you, you: true },
  ...rivals.value.map((r, i) => ({ id: r.id, label: r.label, run: rivalSlots[i]! })),
])

const seed = ref(0)
async function newGame() {
  seed.value = 1 + Math.floor(Math.random() * 1e9)
  const observer = scenario.value.contexts > 1 ? "lamp.v1" : "none.v1" // you can see the lamp: so can they
  await you.start({ scenario: scenario.value.id, observer, strategy: "human", seed: seed.value })
  await Promise.all(
    rivals.value.map((r, i) => rivalSlots[i]!.start({ scenario: scenario.value.id, observer, strategy: r.strategy, params: r.params, seed: seed.value })),
  )
}
onMounted(newGame)
watch(() => scenario.value.id, newGame)

function pull(arm: number) {
  if (you.done.value) return
  you.pull(arm)
  rivalSlots.slice(0, rivals.value.length).forEach((r) => r.step())
  if (you.done.value) finish()
}

const lastScore = ref<number | null>(null)
function finish() {
  const skill = Math.round(you.skill.value * 1000) / 10
  lastScore.value = skill
  if (scenario.value.id !== "classic") return // only the default game is ranked
  const best = loadHumanHistory("bandit").best
  saveHumanGame("bandit", skill)
  emit("score", skill, false, skill >= best ? "You (best)" : "You")
}

// 1-9 (and 0, -, = past nine) pull machines; space or enter starts over once the game is done.
const KEYS = "1234567890-=qwer"
function onKey(event: KeyboardEvent) {
  if (event.target instanceof Element && event.target.closest("input, select, textarea, [contenteditable]")) return
  if (you.done.value && (event.key === " " || event.key === "Enter")) {
    event.preventDefault()
    newGame()
    return
  }
  const arm = KEYS.indexOf(event.key)
  if (arm >= 0 && arm < you.arms.value) {
    event.preventDefault()
    pull(arm)
  }
}
onMounted(() => window.addEventListener("keydown", onKey))
onUnmounted(() => window.removeEventListener("keydown", onKey))

const verdict = computed(() => {
  if (!you.done.value) return null
  const mine = you.skill.value
  const beaten = rivals.value.filter((_, i) => rivalSlots[i]!.skill.value < mine).map((r) => r.label)
  if (mine <= 0.05) return "No better than pulling at random -- the machines kept their secret."
  if (beaten.length === rivals.value.length) return "You beat every strategy on this game."
  if (beaten.length) return `You beat ${beaten.join(" and ")}.`
  return "Every strategy did better on this game."
})
</script>

<template>
  <div class="space-y-5">
    <div class="flex flex-wrap items-end justify-between gap-4">
      <BanditScenarioPicker class="flex-1" />
      <button class="btn-ghost btn-sm" @click="newGame">↻ New game</button>
    </div>

    <BanditFloor :run="you" :scenario="scenario" interactive hide-skill :revealed="you.done.value" @pull="pull" />
    <BanditTape :pulls="you.history.value" :budget="you.budget.value" :binary="scenario.binary" :lamps="scenario.contexts > 1" :drift="you.done.value ? (you.reveal.value?.drift?.at ?? null) : null" />
    <p v-if="!you.done.value" class="text-xs text-fg-subtle">
      Click a machine, or press <kbd class="chip">1</kbd>–<kbd class="chip">{{ Math.min(9, scenario.arms) }}</kbd>. You have
      {{ you.budget.value - you.pulls.value }} pulls left; the strategies below get one pull for each of yours, on the same machines.
    </p>

    <div v-if="you.done.value" class="well flex flex-wrap items-center gap-x-6 gap-y-2 p-4">
      <p class="font-display text-3xl leading-none">Skill {{ lastScore?.toFixed(0) ?? Math.round(you.skill.value * 100) }}</p>
      <p class="min-w-0 flex-1 text-sm text-fg-muted">{{ verdict }}</p>
      <button class="btn-primary btn-sm" @click="newGame">Play again <span class="opacity-60">(space)</span></button>
    </div>

    <div class="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div class="min-w-0">
        <p class="label mb-3">The race · same game, same luck</p>
        <BanditRace :racers="racers" :binary="scenario.binary" :revealed="you.done.value" />
      </div>
      <div v-if="you.done.value && you.reveal.value" class="min-w-0">
        <p class="label mb-2">The reveal</p>
        <BanditReveal :reveal="you.reveal.value" :players="racers.map((r) => ({ label: r.label, counts: r.run.counts.value, you: r.you }))" />
      </div>
    </div>
  </div>
</template>
