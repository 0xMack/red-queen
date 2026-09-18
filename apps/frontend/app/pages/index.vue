<script setup lang="ts">
const highlights = [
  {
    title: "Built from scratch",
    body: "Genetic algorithms, neuroevolution, automatic differentiation, and a transformer -- no ML framework underneath any of it, so every mechanism is something you can actually read.",
  },
  {
    title: "Tested against real games",
    body: "Not toy benchmarks alone -- a policy either learns to eat food and avoid walls in Snake, or it honestly doesn't, and the site says which.",
  },
  {
    title: "Watch it happen, live",
    body: "Every training run streams its progress in real time, and a trained policy plays its game right in the browser -- the same Python code, via Pyodide, not a JS reimplementation.",
  },
]
</script>

<template>
  <main class="mx-auto max-w-4xl px-6 py-12">
    <section class="text-center">
      <h1 class="text-3xl font-bold text-slate-900 sm:text-4xl">Red Queen</h1>
      <p class="mx-auto mt-4 max-w-2xl text-lg text-slate-600">
        A monorepo for exploring reinforcement learning and genetic/evolutionary algorithms via
        custom, from-scratch implementations -- tested against purpose-built games and simulations,
        with real-time visualization to see what's actually happening inside them.
      </p>
      <div class="mt-6 flex justify-center gap-4">
        <NuxtLink
          to="/learn"
          class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Start learning &rarr;
        </NuxtLink>
        <NuxtLink
          to="/games"
          class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Play a game
        </NuxtLink>
      </div>
    </section>

    <section class="mt-16 grid gap-4 sm:grid-cols-3">
      <div v-for="item in highlights" :key="item.title" class="rounded-lg border border-slate-200 bg-white p-5">
        <h2 class="font-semibold text-slate-900">{{ item.title }}</h2>
        <p class="mt-1.5 text-sm text-slate-600">{{ item.body }}</p>
      </div>
    </section>

    <section class="mt-16">
      <h2 class="text-center text-xl font-semibold text-slate-900">
        A policy learning to play Snake, right now, in your browser
      </h2>
      <p class="mx-auto mt-2 max-w-xl text-center text-sm text-slate-500">
        This is the actual game -- click the board, then steer with the arrow keys.
        <NuxtLink to="/learn/teaching-a-snake" class="text-blue-600 hover:underline">
          Here's how it's trained.
        </NuxtLink>
      </p>
      <div class="mt-6 flex justify-center">
        <LiveSnakeDemo />
      </div>
    </section>

    <section class="mt-16 rounded-lg border border-slate-200 bg-white p-6 text-center">
      <h2 class="font-semibold text-slate-900">See it training</h2>
      <p class="mt-1.5 text-sm text-slate-600">
        Every run -- genetic programming, neuroevolution -- streams its progress live.
      </p>
      <NuxtLink to="/runs" class="mt-3 inline-block text-sm font-medium text-blue-600 hover:underline">
        Browse training runs &rarr;
      </NuxtLink>
    </section>
  </main>
</template>
