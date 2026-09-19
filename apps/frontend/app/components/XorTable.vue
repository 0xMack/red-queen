<script setup lang="ts">
// The four XOR rows with what a network actually outputs for each -- the ground truth every Learn demo on
// this page family is judged against. `outputs` are tanh values in [-1, 1]; shown mapped to [0, 1] (the
// scale the targets live on) so "0.93" reads as "probably 1". `active` highlights one row (the one whose
// activations a neighbouring diagram is showing); rows are clickable when `selectable`.
import { XOR_CASES } from "~/utils/neat"

const props = withDefaults(
  defineProps<{ outputs: number[]; active?: number | null; selectable?: boolean; label?: string }>(),
  { active: null, selectable: false, label: "network says" },
)
const emit = defineEmits<{ select: [row: number] }>()

// Rounded: these become style/attribute strings, and Node and the browser can disagree in the last digit of
// Math.tanh -- an unrounded value is a hydration mismatch waiting to happen.
const rows = computed(() =>
  XOR_CASES.map((c, i) => {
    const value = Math.round(((props.outputs[i]! + 1) / 2) * 1000) / 1000
    return { ...c, value, right: Math.abs(value - c.target) < 0.25 }
  }),
)
</script>

<template>
  <table class="w-full border-separate border-spacing-y-1 text-xs">
    <thead>
      <tr class="text-left text-[10px] tracking-wide text-fg-subtle uppercase">
        <th class="pl-2 font-medium">x₀ x₁</th>
        <th class="font-medium">XOR</th>
        <th class="font-medium">{{ label }}</th>
        <th class="pr-2 text-right font-medium" />
      </tr>
    </thead>
    <tbody>
      <tr
        v-for="(r, i) in rows"
        :key="i"
        class="rounded-md bg-sunken transition"
        :class="[selectable ? 'cursor-pointer hover:bg-raised' : '', active === i ? 'outline outline-1 outline-gold-400/60' : '']"
        @click="selectable && emit('select', i)"
      >
        <td class="num rounded-l-md py-1.5 pl-2 text-fg-muted">{{ r.input[0] }} {{ r.input[1] }}</td>
        <td class="num py-1.5 text-fg">{{ r.target }}</td>
        <td class="py-1.5">
          <span class="num text-fg">{{ r.value.toFixed(2) }}</span>
          <span class="ml-2 inline-block h-1.5 w-16 rounded-full bg-line align-middle">
            <span class="block h-full rounded-full" :class="r.right ? 'bg-life-400' : 'bg-queen-400'" :style="{ width: `${Math.max(0, Math.min(1, r.value)) * 100}%` }" />
          </span>
        </td>
        <td class="rounded-r-md py-1.5 pr-2 text-right" :class="r.right ? 'text-life-300' : 'text-queen-300'">{{ r.right ? "✓" : "✗" }}</td>
      </tr>
    </tbody>
  </table>
</template>
