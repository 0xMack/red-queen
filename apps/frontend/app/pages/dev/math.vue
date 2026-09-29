<script setup lang="ts">
import * as bandits from "~/data/math/multi-armed-bandits"
import { SYMBOLS } from "~/data/math/symbols"
import { termsOfFormula, type Formula } from "~/utils/math/expr"

// Every formula in data/math/, rendered with its terms (docs/design/0012) -- the place to check a new formula
// compiles (KaTeX throws in development) and every term reads right, and to try a lab's formula at the top of a page.
// Unlinked, like /dev/rl. `?lab=ucb` (or any BanditLab `math` value) puts that lab first; `&scenario=` picks its game.
useHead({ title: "Formulas" })
const route = useRoute()
const lab = computed(() => (typeof route.query.lab === "string" ? route.query.lab : null))
// `&scenario=lucky-start` plays the lab on another scenario (Gaussian payouts: negative numbers, two decimals).
const scenario = computed(() => (typeof route.query.scenario === "string" ? route.query.scenario : lab.value === "bellman" ? "detour" : "classic"))
const groups: { chapter: string; formulas: Formula[] }[] = [{ chapter: "multi-armed-bandits", formulas: Object.values(bandits) }]
const unknownSymbols = groups.flatMap((g) =>
  g.formulas.flatMap((f) => termsOfFormula(f).filter((t) => t.symbol && !(t.symbol in SYMBOLS)).map((t) => `${f.id}: ${t.id} → ${t.symbol}`)),
)
</script>

<template>
  <main class="mx-auto max-w-4xl px-4 py-10">
    <h1 class="text-3xl">Formulas</h1>
    <p class="mt-2 text-sm text-fg-muted">docs/design/0012. Hover or click any term.</p>
    <p v-if="unknownSymbols.length" class="mt-2 text-sm text-queen-300">Terms naming unknown symbols: {{ unknownSymbols.join(", ") }}</p>

    <BanditLab v-if="lab" class="mt-6" :title="`Lab: ${lab}`" :scenarios="[scenario]" :gamma="lab === 'bellman'" :math="lab as any" />

    <section v-for="g in groups" :key="g.chapter" class="mt-8">
      <p class="label">{{ g.chapter }}</p>
      <div v-for="f in g.formulas" :key="f.id">
        <MathScope>
          <MathFormula :formula="f" :caption="f.id" />
        </MathScope>
      </div>
    </section>
  </main>
</template>
