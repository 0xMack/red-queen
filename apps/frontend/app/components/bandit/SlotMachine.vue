<script setup lang="ts">
// One machine on the bandit floor (docs/design/0011): an arched cabinet with its letter and a marquee of bulbs in the
// machine's colour (they chase on a win), a glass reel that spins when the machine is pulled and lands on the payout,
// a lever, and a coin tray holding this machine's last few outcomes. With `estimate` it also shows what a strategy
// believes it pays (a bar, and a band for how unsure it is); with `trueMean` -- only after the game -- where the truth
// was. Presentational: the owner decides what a pull means.
const props = withDefaults(
  defineProps<{
    arm: number
    /** Increments on every pull of *this* machine: restarts the spin. */
    spin: number
    /** Its last payout, or null before it has been pulled. */
    reward: number | null
    /** Its last few payouts, oldest first (the coin tray). */
    recent?: number[]
    binary: boolean
    pulls: number
    /** Spin length (ms); under ~60 the reel just shows the result. */
    spinMs?: number
    interactive?: boolean
    /** A strategy has picked this machine for the next pull. */
    chosen?: boolean
    /** The belief overlay: estimate and spread on a 0..scale bar. */
    estimate?: number | null
    spread?: number | null
    /** Probability the strategy pulls it next (a strategy that chooses by chance). */
    probability?: number | null
    scale?: number
    /** The reveal. */
    trueMean?: number | null
    best?: boolean
    compact?: boolean
  }>(),
  {
    recent: () => [],
    spinMs: 360,
    interactive: false,
    chosen: false,
    estimate: null,
    spread: null,
    probability: null,
    scale: 1,
    trueMean: null,
    best: false,
    compact: false,
  },
)
const emit = defineEmits<{ pull: [arm: number] }>()

const color = computed(() => armColor(props.arm))
const name = computed(() => armName(props.arm))

// The spin: a strip of symbols scrolls, blurred, then the payout lands. Restarted by `spin` changing.
const spinning = ref(false)
let timer: ReturnType<typeof setTimeout> | null = null
watch(
  () => props.spin,
  () => {
    if (props.spinMs < 60) return
    spinning.value = false
    requestAnimationFrame(() => {
      spinning.value = true
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => (spinning.value = false), props.spinMs)
    })
  },
)
onUnmounted(() => timer && clearTimeout(timer))

const SYMBOLS = ["♛", "7", "◆", "★", "♣", "●"]
const isWin = (r: number) => (props.binary ? r >= 1 : r > 0)
const shown = computed(() => (props.reward === null ? "" : formatPayout(props.reward, props.binary)))
const won = computed(() => props.reward !== null && isWin(props.reward) && !spinning.value && props.spin > 0)
const bulbs = computed(() => (props.compact ? 5 : 7))

const pct = (v: number) => `${Math.max(0, Math.min(100, (v / props.scale) * 100))}%`
</script>

<template>
  <button
    type="button"
    class="machine group relative flex min-w-0 flex-col items-stretch border text-left transition duration-150"
    :class="[
      interactive ? 'cursor-pointer hover:-translate-y-0.5 active:translate-y-0' : 'cursor-default',
      chosen ? 'is-chosen' : best ? 'is-best' : '',
      compact ? 'rounded-t-[16px] rounded-b-[7px] px-1.5 pt-1.5 pb-1.5' : 'rounded-t-[26px] rounded-b-[9px] px-2.5 pt-2 pb-2.5',
    ]"
    :style="{ '--arm': color }"
    :disabled="!interactive"
    :aria-label="`Machine ${name}${pulls ? `, pulled ${pulls} times` : ''}`"
    @click="interactive && emit('pull', arm)"
  >
    <!-- Marquee: bulbs in the machine's colour that chase on a win, and its letter -->
    <div class="marquee flex justify-center gap-1" :class="{ winning: won }" :key="`m${spin}`">
      <span v-for="b in bulbs" :key="b" class="bulb rounded-full" :class="compact ? 'size-1' : 'size-1.5'" :style="{ animationDelay: `${b * 60}ms` }" />
    </div>
    <p class="text-center font-display leading-none" :class="compact ? 'mt-1 text-lg' : 'mt-1.5 text-[1.7rem]'">{{ name }}</p>

    <!-- The reel, behind glass, with a payline; and the lever -->
    <div class="mt-1.5 flex items-center gap-1">
      <div class="reel relative min-w-0 flex-1 overflow-hidden rounded-[6px]" :class="compact ? 'h-8' : 'h-12'">
        <div v-if="spinning" class="strip absolute inset-x-0 top-0 flex flex-col items-center" :style="{ animationDuration: `${Math.max(90, spinMs / 3)}ms` }">
          <span v-for="(s, i) in [...SYMBOLS, ...SYMBOLS]" :key="i" class="flex items-center justify-center text-fg-subtle" :class="compact ? 'h-8 text-sm' : 'h-12 text-lg'">{{ s }}</span>
        </div>
        <div
          v-else
          class="result num absolute inset-0 flex items-center justify-center font-semibold tracking-wide"
          :class="[won ? 'text-[var(--arm)]' : 'text-fg-subtle', compact ? 'text-[11px]' : 'text-[15px]']"
          :key="`r${spin}`"
        >
          {{ shown || "·" }}
        </div>
        <span class="payline pointer-events-none absolute inset-x-1 top-1/2 h-px" />
        <span class="glass pointer-events-none absolute inset-0" />
      </div>
      <svg v-if="!compact" class="lever h-12 w-2.5 shrink-0" viewBox="0 0 10 48" aria-hidden="true" :class="{ pulled: spinning }">
        <rect x="3.5" y="10" width="3" height="34" rx="1.5" :fill="palette.lineStrong" />
        <circle cx="5" cy="7" r="4.5" :fill="color" />
        <rect x="1" y="42" width="8" height="5" rx="1.5" :fill="palette.line" />
      </svg>
    </div>

    <!-- Coin tray: this machine's last outcomes, and how often it's been pulled -->
    <div class="tray mt-1.5 flex items-center gap-1 rounded-[5px] px-1.5" :class="compact ? 'h-4' : 'h-5'">
      <span
        v-for="(r, i) in recent"
        :key="i"
        class="coin shrink-0 rounded-full"
        :class="[compact ? 'size-1' : 'size-1.5', isWin(r) ? 'win' : '']"
        :title="formatPayout(r, binary)"
      />
      <span class="num ml-auto text-[10px] text-fg-subtle">{{ pulls }}×</span>
    </div>

    <!-- What the strategy believes; where the truth was -->
    <div v-if="estimate !== null || trueMean !== null" class="relative mt-2 h-1.5 rounded-full bg-sunken" :title="estimate !== null ? `believes ${estimate.toFixed(2)}` : undefined">
      <div v-if="estimate !== null && spread" class="absolute inset-y-0 rounded-full bg-[var(--arm)] opacity-25" :style="{ left: pct(estimate - spread), width: `calc(${pct(estimate + spread)} - ${pct(estimate - spread)})` }" />
      <div v-if="estimate !== null" class="absolute inset-y-0 left-0 rounded-full bg-[var(--arm)] transition-[width] duration-200" :style="{ width: pct(estimate) }" />
      <div v-if="trueMean !== null" class="absolute -top-1 h-3.5 w-0.5 rounded-full bg-gold-300" :style="{ left: pct(trueMean) }" :title="`true value ${trueMean.toFixed(2)}`" />
    </div>
    <p v-if="probability !== null" class="mt-1 text-right font-mono text-[10px] text-fg-subtle">{{ Math.round(probability * 100) }}% next</p>
  </button>
</template>

<style scoped>
.machine {
  border-color: var(--color-line);
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--arm) 9%, transparent), transparent 38%),
    var(--color-surface);
  box-shadow: inset 0 1px 0 rgb(255 255 255 / 0.04);
}
.machine:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--arm) 45%, var(--color-line-strong));
}
.machine.is-chosen {
  border-color: var(--color-fg);
  box-shadow: 0 0 0 1px var(--color-fg);
}
.machine.is-best {
  border-color: var(--color-gold-400);
}
.bulb {
  background: var(--arm);
  opacity: 0.28;
}
.marquee.winning .bulb {
  animation: chase 650ms ease-out 2;
}
@keyframes chase {
  0%,
  100% {
    opacity: 0.28;
    box-shadow: none;
  }
  40% {
    opacity: 1;
    box-shadow: 0 0 6px 1px var(--arm);
  }
}
.reel {
  background: linear-gradient(180deg, #050504, var(--color-sunken) 45%, #050504);
  border: 1px solid var(--color-line);
  box-shadow: inset 0 2px 6px rgb(0 0 0 / 0.7);
}
.glass {
  background: linear-gradient(180deg, rgb(255 255 255 / 0.06), transparent 40%);
}
.payline {
  background: color-mix(in srgb, var(--color-queen-400) 40%, transparent);
}
.strip {
  animation-name: roll;
  animation-timing-function: linear;
  animation-iteration-count: infinite;
  filter: blur(1.2px);
}
@keyframes roll {
  from {
    transform: translateY(-50%);
  }
  to {
    transform: translateY(0);
  }
}
.result {
  animation: land 220ms cubic-bezier(0.2, 1.4, 0.4, 1);
}
@keyframes land {
  from {
    transform: translateY(-40%);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}
.lever {
  transform-origin: 50% 88%;
  transition: transform 160ms ease-out;
}
.lever.pulled {
  transform: scaleY(-0.6);
}
.machine:hover:not(:disabled) .lever {
  transform: rotate(10deg);
}
.tray {
  background: var(--color-sunken);
  box-shadow: inset 0 1px 3px rgb(0 0 0 / 0.6);
}
.coin {
  border: 1px solid var(--color-line-strong);
}
.coin.win {
  border-color: var(--arm);
  background: var(--arm);
}
@media (prefers-reduced-motion: reduce) {
  .strip,
  .result,
  .marquee.winning .bulb {
    animation: none;
  }
}
</style>
