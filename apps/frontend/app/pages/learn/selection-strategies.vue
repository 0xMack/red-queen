<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import * as code from "~/data/snippets/selection-strategies"
import * as math from "~/data/math/selection-strategies"

const paretoUsage = `selection = ParetoSelection(
    complexity=lambda program: program.effective_instruction_count(),
    k=3,
)`
</script>

<template>
  <article class="prose-chapter">
    <p>
      "Select parents biased toward fitness" sounds simple until you ask: biased <em>how</em>?
      The obvious answer -- average each individual's performance across every test case, favor
      the best average -- is <strong>Tournament Selection</strong>, and it's a perfectly
      reasonable default. It's also not the only answer, and the alternatives exist because
      tournament selection has a real, specific failure mode.
    </p>

    <h2>Tournament Selection</h2>
    <p>
      Pick <code>k</code> individuals at random,
      return whichever has the best mean fitness across all test cases. Simple, fast, and it works:
    </p>
    <MathScope>
      <MathFormula :formula="math.mean" />
      <MathFormula :formula="math.tournament" />
    </MathScope>
    <CodeBlock :snippet="code.tournament" />

    <Callout variant="warning" title="The problem with averaging">
      A population selected purely on mean fitness can get <em>better on average</em> while a
      specific champion gets <em>worse on a specific case</em> than an earlier, weaker-on-average
      champion was. This isn't a bug -- it's a real, unavoidable property of optimizing an
      aggregate. Scaling up neuroevolution training on this project's Snake game showed exactly
      this: mean fitness rose and a total-failure scenario got fixed, but one previously-fine
      scenario got worse in the process. "Better on average" doesn't mean "better everywhere."
    </Callout>

    <h2>Lexicase Selection</h2>
    <p>
      Lexicase selection never averages. It filters candidates one test case at a time, in random
      order each time it's called, keeping only whoever's within a small tolerance of the best
      performer on that case -- until one candidate survives or every case has been used:
    </p>
    <MathScope>
      <MathFormula
        :formula="math.lexicaseCut"
        :values="{ best: 0.9, 'epsilon-c': 0.1 }"
        caption="An example case: four candidates score 0.9, 0.8, 0.7 and 0.4. The median is 0.75, the distances from it are 0.15, 0.05, 0.05 and 0.35, so ε is their median, 0.10 -- and 0.9 and 0.8 survive."
      />
      <MathFormula :formula="math.lexicaseEpsilon" />
    </MathScope>
    <CodeBlock :snippet="code.lexicase" />

    <Callout variant="finding" title="A specialist can beat a generalist">
      Comparing tournament and lexicase selection on the same fixed benchmark
      (<code class="text-xs">notebooks/0001-tournament-vs-lexicase.ipynb</code>), lexicase
      sometimes preferred a "specialist" -- excellent on a subset of cases, mediocre elsewhere --
      over a "generalist" with a better overall average. That's the point, not a defect: if some
      cases are much harder than others, an aggregate score can bury the individual that's
      actually closest to solving the hard ones.
    </Callout>

    <h2>Pareto Selection</h2>
    <p>
      Tournament and lexicase both answer "which individual is better?" using fitness alone.
      Pareto selection asks a different question: is this individual better <em>and</em> simpler
      than that one? It optimizes a genuine tradeoff curve (accuracy vs. an injected complexity
      measure) rather than a single "best":
    </p>
    <MathFormula :formula="math.dominance" caption="Within a tournament, the winner is drawn from the genomes nothing else dominates." />
    <CodeBlock lang="python" :code="paretoUsage" />

    <Callout variant="finding" title="A real tradeoff, not a free lunch">
      Running Pareto selection against the same benchmark
      (<code class="text-xs">notebooks/0002-pareto-selection.ipynb</code>) produced smaller
      programs, at a real accuracy cost compared to tournament selection on that benchmark --
      reported honestly as a tradeoff, not tuned until it looked like a strict win. Simpler isn't
      free.
    </Callout>

    <p>
      All three strategies are <strong>genome-generic</strong> -- they only ever look at fitness
      values, never at what a genome actually is. That's what lets the exact same
      <code>TournamentSelection</code>/<code>LexicaseSelection</code>
      classes work unchanged whether the population is register-machine programs, expression
      trees, or a neural network's weights --
      <NuxtLink to="/learn/teaching-a-snake">the Snake case study</NuxtLink>
      uses <code class="text-xs">LexicaseSelection</code> against a neuroevolved policy, the same
      class shown above, with zero changes.
    </p>
  </article>
</template>
