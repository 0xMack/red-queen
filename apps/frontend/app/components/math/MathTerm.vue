<script setup lang="ts">
import { mathSymbol } from "~/data/math/symbols"
import { renderTex } from "~/utils/math/katex"
import "katex/dist/katex.min.css"

// A term named in prose -- "a step of <MathTerm id="alpha" />" -- typeset like the formula's, and part of the nearest
// term scope: hovering it lights the term everywhere in the figure, and the other way round. `id` is the term id in
// the scope; `symbol` (default: the id) says how to write it, or pass `tex`.
const props = defineProps<{ id: string; symbol?: string; tex?: string }>()
const scope = useTermScope()
const html = computed(() => renderTex(props.tex ?? mathSymbol(props.symbol ?? props.id)?.tex ?? props.id, false))
</script>

<template>
  <span class="math-formula math-term inline-block" v-bind="scope.target(id)" tabindex="0">
    <span data-term-inline v-html="html" />
  </span>
</template>

<style scoped>
.math-term.term-on {
  color: var(--term-color);
  background-color: color-mix(in srgb, var(--term-color) 14%, transparent);
  outline-color: transparent;
}
</style>
