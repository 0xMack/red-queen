<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
const baselineCode = `from evolve import (
    LinearCrossoverMutation,
    SymbolicRegressionFitness,
    TournamentSelection,
    evolve,
    random_program,
)

rng = random.Random(0)
population = [
    random_program(num_instructions=12, num_registers=4, num_inputs=1, rng=rng)
    for _ in range(60)
]

evolve(
    population,
    fitness=SymbolicRegressionFitness(
        target=lambda x: x**4 - 3 * x**2 + 2,
        inputs=[i / 5 for i in range(-5, 6)],
    ),
    selection=TournamentSelection(k=3),
    variation=LinearCrossoverMutation(mutation_rate=0.1),
    generations=60,
    on_generation=[lambda s: print(s.generation, s.best_fitness)],
    rng=rng,
)`

const summaryCode = `class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    champion: Genome`

const runId = ref<string | null>(null)
const api = useApi()
onMounted(async () => {
  try {
    const runs = await api.fetch<{ run_id: string; config?: Record<string, unknown> }[]>("/runs")
    runId.value = runs.find((r) => r.config?.representation === "linear_gp")?.run_id ?? null
  } catch {
    // No backend reachable -- the chapter reads fine without the live chart, it just won't show.
  }
})
</script>

<template>
  <article class="prose-chapter">
    <p>
      A genetic algorithm evolves a <strong>population</strong> of candidate solutions toward a goal
      it can only measure, never differentiate. There's no gradient telling it which direction to
      move -- just a score, per candidate, called <strong>fitness</strong>. Everything else is a
      loop around that one fact:
    </p>

    <ol class="ml-5 list-decimal space-y-1">
      <li>Score every individual in the population against the fitness function.</li>
      <li>Select parents, biased toward higher fitness (but not exclusively the best -- more on why in the next chapter).</li>
      <li>Vary them (mutation, sometimes crossover) to produce the next generation.</li>
      <li>Repeat.</li>
    </ol>

    <p>
      That's the whole algorithm. What makes it interesting is what you plug into each of those
      four steps -- a genome can be a register-machine program, an expression tree, or a neural
      network's weights (later chapters cover all three), and the loop itself doesn't change.
    </p>

    <h2>A concrete run</h2>
    <p>
      This project's baseline example evolves a linear-genetic-programming individual -- a small
      register-machine program -- to approximate <code>x⁴ - 3x² + 2</code>
      from 11 sample points, using nothing but add/subtract/multiply/divide:
    </p>

    <CodeBlock lang="python" :code="baselineCode" />

    <p>
      <code>on_generation</code> is the one seam
      that matters architecturally: <code>evolve()</code>
      itself has zero knowledge of how (or whether) a run gets recorded anywhere. It just calls
      every callback in that list once per generation with a plain summary object:
    </p>

    <CodeBlock lang="python" :code="summaryCode" />

    <Callout variant="note" title="Why this matters">
      This is what makes the live charts on this site possible without coupling the algorithm to
      telemetry, a database, or a web framework. A different callback adapts that summary into a
      recorded metric; <code class="text-xs">evolve()</code> never imports any of it. You could
      run this exact loop in a plain script with no callbacks at all and it would behave
      identically.
    </Callout>

    <h2>What that actually looks like</h2>
    <p v-if="!runId" class="text-sm text-fg-subtle">
      (Live chart unavailable -- start the backend to see a real run's fitness curve here.)
    </p>
    <ClientOnly v-else>
      <RunFitnessPreview :run-id="runId" />
    </ClientOnly>

    <p>
      Best fitness climbs, generation over generation, as fitter programs get selected more often
      and their variations occasionally do better still. It's not smooth or monotonic -- variation
      is random, so some generations regress -- but the trend is real.
    </p>

    <p>
      Next: <NuxtLink to="/learn/selection-strategies">how "select parents biased toward fitness" actually happens</NuxtLink>,
      and why the obvious way to do it (average fitness, pick the best) isn't always the right one.
    </p>
  </article>
</template>
