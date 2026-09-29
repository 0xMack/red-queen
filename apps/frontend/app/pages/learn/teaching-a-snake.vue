<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import * as code from "~/data/snippets/teaching-a-snake"

const trainingCode = `LAYER_SIZES = (11, 16, 3)  # was (100, 24, 3) -- ~10x fewer weights

evolve(
    population,
    fitness=SimulationFitnessEvaluator(
        envs=benchmark_environments(), act=act, max_steps=200
    ),
    selection=LexicaseSelection(),  # was TournamentSelection(k=4)
    variation=GaussianMutation(sigma=0.2),
    generations=250,
    on_generation=[...],
    rng=rng,
)`

</script>

<template>
  <article class="prose-chapter">
    <p>
      Neuroevolution replaces a genetic algorithm's genome with a neural network's weights and
      evolves them directly -- no backpropagation, no gradients, just mutation and selection
      applied to a real (small) network. It's Evolution Strategies, and it's how the policy below
      was trained. Play it first, then read how it got this way:
    </p>

    <figure class="card my-8 grid items-center gap-6 p-5 sm:grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)]">
      <LiveSnakeDemo class="mx-auto" />
      <figcaption class="text-sm text-fg-muted">
        <p class="font-display text-fg">Your turn first</p>
        <p class="mt-2">
          Click the board and steer with the arrow keys. This is the real Snake game core, compiled to
          WebAssembly -- the same code every policy below is evaluated on.
        </p>
        <p class="mt-3">
          Then <NuxtLink to="/runs">open a Snake run</NuxtLink> to watch the evolved champion play,
          with its network lighting up move by move.
        </p>
      </figcaption>
    </figure>

    <h2>Attempt one: show it the whole board</h2>
    <p>
      The first version handed the network the entire board, flattened -- one float per cell,
      100 numbers on a 10x10 grid:
    </p>
    <CodeBlock :snippet="code.oldObservation" />
    <p>
      It trained. It even improved, generation over generation. But after 150 generations it
      plateaued at a modest result, and scaling up to a much bigger population and generation
      budget helped only a little.
    </p>

    <Callout variant="finding" title="A representation ceiling, not a compute shortage">
      Diversity stayed healthy and the population never collapsed -- ruling out the usual "ran out
      of exploration" explanation. The real problem: a flat network with no spatial prior has to
      <em>re-discover</em> "is there a wall two cells ahead?" and "which way is the food?" from
      raw cell values, using nothing but evolutionary search, before it can even start learning
      good decisions from that structure. With a fixed search budget, most of it was going toward
      rediscovering structure a human would notice instantly.
    </Callout>

    <h2>Attempt two: hand it the structure directly</h2>
    <p>
      The fix: stop making the network infer structure, and hand it the structure instead. 11
      features -- is there danger immediately ahead/left/right, which way is the snake currently
      heading, which way is the food:
    </p>
    <CodeBlock :snippet="code.newObservation" />

    <p>
      Combined with switching from tournament to
      <NuxtLink to="/learn/selection-strategies">lexicase selection</NuxtLink>
      (Snake's fitness is 5 genuinely different benchmark scenarios -- exactly the case lexicase is
      for) and a smaller, ~10x-fewer-parameter network that trains faster as a direct result:
    </p>
    <CodeBlock lang="python" :code="trainingCode" />

    <Callout variant="finding" title="best_fitness: 0.65 → 17.28">
      Same generation-count class, same benchmark, same reward function -- only the observation
      and selection strategy changed. The retrained policy doesn't just survive longer, it
      actually eats food repeatedly in a single episode. The live demo at the top of this page is
      running that result.
    </Callout>

    <h2>A second bug, found the same way: by running it</h2>
    <p>
      Before either observation redesign, an earlier version of the reward function gave a
      symmetric nudge toward/away from food. An evolved policy found a loophole:
    </p>
    <CodeBlock :snippet="code.rewardShaping" />
    <Callout variant="warning" title="Reward hacking, not a training failure">
      With a symmetric reward, a policy that oscillates between two adjacent cells forever nets
      ~0 reward -- a real, lower-risk local optimum than continuing to risk death by seeking food.
      Making the "farther" penalty larger than the "closer" reward makes standing still strictly
      worse than seeking the goal, not merely no-better. Neither bug -- this one or the
      observation ceiling above -- was something anyone guessed at in advance. Both were found by
      actually training a policy and watching what it learned to do.
    </Callout>

    <p>
      This project's whole approach to reinforcement learning and genetic algorithms comes down to
      that last sentence: build the mechanism, run it for real, and report what actually happened
      -- including the parts that didn't work the first time.
    </p>
  </article>
</template>
