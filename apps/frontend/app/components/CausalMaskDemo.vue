<script setup lang="ts">
// The causal mask from libs/tinylm/src/tinylm/layers.py (_causal_mask): position i may attend to
// positions 0..i and nothing after -- the model predicts each next character without seeing it.
// Structure only (which cells are allowed), which is exact; no invented attention weights.
const text = ref("Alice was ")
const chars = computed(() => [...text.value.slice(0, 16)])
const hovered = ref<number>(4)
const show = (c: string) => (c === " " ? "␣" : c)
</script>

<template>
  <figure class="card my-8 p-5">
    <div class="flex flex-wrap items-center gap-3 text-xs">
      <label class="text-fg-subtle" for="mask-text">sequence</label>
      <input
        id="mask-text"
        v-model="text"
        maxlength="16"
        class="w-48 rounded-md border border-line bg-sunken px-2 py-1 font-mono text-fg focus:border-queen-400/60 focus:outline-none"
      >
      <span class="text-fg-subtle">hover a row</span>
    </div>
    <div class="mt-4 overflow-x-auto">
      <table class="border-separate border-spacing-0.5 font-mono text-[11px]">
        <thead>
          <tr>
            <th />
            <th v-for="(c, j) in chars" :key="j" class="w-6 pb-1 font-normal" :class="j <= hovered ? 'text-fg' : 'text-fg-subtle'">{{ show(c) }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(c, i) in chars" :key="i" @mouseenter="hovered = i">
            <th class="pr-2 text-right font-normal" :class="i === hovered ? 'text-queen-300' : 'text-fg-subtle'">{{ show(c) }}</th>
            <td
              v-for="(_, j) in chars"
              :key="j"
              class="size-6 rounded-sm transition-colors"
              :class="
                j > i
                  ? 'bg-sunken'
                  : i === hovered
                    ? 'bg-queen-400'
                    : 'bg-queen-500/25'
              "
            />
          </tr>
        </tbody>
      </table>
    </div>
    <p class="mt-3 text-sm text-fg-muted">
      Predicting what comes after <span class="font-mono text-queen-300">“{{ chars.slice(0, hovered + 1).join("") }}”</span>, the
      character at position {{ hovered }} can look at
      <span class="num text-fg">{{ hovered + 1 }}</span> position{{ hovered ? "s" : "" }} -- itself and everything before it.
      Everything to its right is set to −10⁹ before the softmax, so it gets zero weight.
    </p>
  </figure>
</template>
