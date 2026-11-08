<script setup lang="ts">
import * as math from "~/data/math/tracing-images"

// Why a near-circle becomes a circle: a circle is cheap to write down but may get some pixels wrong, four cubics draw the
// outline exactly but cost far more to say. The bits are `Costs::model_bits` at 128 px: 3 for a type, log2(1280) + 1 for
// a coordinate, 12 for a wrong pixel.
const COORD = Math.log2(1280) + 1
const TYPE = 3
const PIXEL = 12
const CIRCLE = TYPE + 3 * COORD // a type, a centre and a radius
const PATH = TYPE * 7 + 26 * COORD // a move, four cubics and a close, plus the path's own type: 6 segments, 26 numbers

const wrong = ref(10)
const circle = computed(() => CIRCLE + PIXEL * wrong.value)
const choice = computed(() => (circle.value <= PATH ? 0 : 1))
const tie = Math.floor((PATH - CIRCLE) / PIXEL)
</script>

<template>
  <UiFigure title="A near-circle: three numbers, or twenty-six?">
    <div class="space-y-4">
      <UiRange v-model="wrong" label="Pixels the circle gets wrong (the outline gets none)" :min="0" :max="60" />
      <MathFormula :formula="math.length" :values="{ model: Math.round(CIRCLE), pixel: PIXEL, wrong }" caption="The circle's description." />
      <UiBars
        :labels="['circle', 'Bézier path']"
        :values="[circle, PATH]"
        :chosen="choice"
        :max="PIXEL * 60 + CIRCLE"
        :format="(v) => `${Math.round(v)} bits`"
      />
    </div>
    <template #caption>
      The circle wins until it gets about {{ tie }} pixels wrong. That margin is why a slightly dented circle in a noisy image
      comes back as a circle, and why a shape that really isn't one doesn't.
    </template>
  </UiFigure>
</template>
