<script setup lang="ts">
// A one-neuron computation graph, L = (tanh(x·w + b) − t)², with every node's forward value and
// backward gradient shown live -- the same mechanics libs/autodiff's Tensor runs (each op records its
// parents and a local backward rule; backward() walks the graph in reverse topological order and
// accumulates grad = upstream × local derivative). Mirrors the engine's rules for *, +, tanh, -,
// **2 exactly; checked against a central finite difference, the same way libs/autodiff/tests do.
const x = ref(1.5)
const w = ref(-0.8)
const b = ref(0.3)
const t = ref(0.5)
const lr = 0.25
// Built here, not inline in the template: the template auto-unwraps top-level refs, so an inline
// [x, w, ...] would hand each slider a plain number instead of the ref it needs to write to.
const sliders = [
  { name: "x (input)", model: x, min: -2, max: 2 },
  { name: "w (weight)", model: w, min: -2, max: 2 },
  { name: "b (bias)", model: b, min: -2, max: 2 },
  { name: "t (target)", model: t, min: -1, max: 1 },
]
const history = ref<number[]>([])

const graph = computed(() => {
  // forward
  const u = x.value * w.value
  const z = u + b.value
  const a = Math.tanh(z)
  const e = a - t.value
  const L = e * e
  // backward: seed dL/dL = 1, then each node's local rule
  const dL = 1
  const de = 2 * e * dL // **2
  const da = de // a - t
  const dz = (1 - a * a) * da // tanh
  const du = dz // u + b
  const db = dz
  const dx = w.value * du // x * w
  const dw = x.value * du
  return { u, z, a, e, L, grads: { L: dL, e: de, a: da, z: dz, u: du, b: db, x: dx, w: dw } }
})

// Central finite difference for dL/dw -- what the engine's gradient must match.
const numericalDw = computed(() => {
  const f = (wv: number) => (Math.tanh(x.value * wv + b.value) - t.value) ** 2
  const eps = 1e-6
  return (f(w.value + eps) - f(w.value - eps)) / (2 * eps)
})

function step() {
  history.value = [...history.value.slice(-19), graph.value.L]
  w.value -= lr * graph.value.grads.w
  b.value -= lr * graph.value.grads.b
}
function reset() {
  x.value = 1.5
  w.value = -0.8
  b.value = 0.3
  t.value = 0.5
  history.value = []
}

const nodes = computed(() => {
  const g = graph.value
  return [
    { id: "x", label: "x", op: "input", value: x.value, grad: g.grads.x, cx: 60, cy: 50 },
    { id: "w", label: "w", op: "param", value: w.value, grad: g.grads.w, cx: 60, cy: 150 },
    { id: "b", label: "b", op: "param", value: b.value, grad: g.grads.b, cx: 60, cy: 250 },
    { id: "u", label: "×", op: "u = x·w", value: g.u, grad: g.grads.u, cx: 210, cy: 100 },
    { id: "z", label: "+", op: "z = u + b", value: g.z, grad: g.grads.z, cx: 340, cy: 170 },
    { id: "a", label: "tanh", op: "a = tanh z", value: g.a, grad: g.grads.a, cx: 470, cy: 170 },
    { id: "e", label: "−", op: "e = a − t", value: g.e, grad: g.grads.e, cx: 600, cy: 170 },
    { id: "L", label: "²", op: "L = e²", value: g.L, grad: g.grads.L, cx: 730, cy: 170 },
  ]
})
const EDGES: [string, string][] = [["x", "u"], ["w", "u"], ["u", "z"], ["b", "z"], ["z", "a"], ["a", "e"], ["e", "L"]]
const byId = computed(() => Object.fromEntries(nodes.value.map((n) => [n.id, n])))
const fmt = (v: number) => (Math.abs(v) < 1e-4 ? "0.000" : v.toFixed(3))
</script>

<template>
  <figure class="card my-8 p-5">
    <svg viewBox="0 0 800 300" class="block h-auto w-full">
      <defs>
        <marker id="ad-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#323a4d" />
        </marker>
      </defs>
      <line
        v-for="[from, to] in EDGES"
        :key="`${from}-${to}`"
        :x1="byId[from]!.cx + 28"
        :y1="byId[from]!.cy"
        :x2="byId[to]!.cx - 30"
        :y2="byId[to]!.cy"
        stroke="#323a4d"
        stroke-width="1.5"
        marker-end="url(#ad-arrow)"
      />
      <g v-for="n in nodes" :key="n.id">
        <circle :cx="n.cx" :cy="n.cy" r="26" :fill="n.op === 'param' ? '#1d1420' : '#151924'" :stroke="n.op === 'param' ? '#ff5c7a' : '#60a5fa'" stroke-width="1.5" />
        <text :x="n.cx" :y="n.cy + 5" text-anchor="middle" class="fill-fg font-mono text-[14px]">{{ n.label }}</text>
        <text :x="n.cx" :y="n.cy - 36" text-anchor="middle" class="num fill-signal-300 text-[11px]">{{ fmt(n.value) }}</text>
        <text :x="n.cx" :y="n.cy + 44" text-anchor="middle" class="num fill-queen-300 text-[11px]">∂L {{ fmt(n.grad) }}</text>
      </g>
    </svg>

    <div class="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div class="grid grid-cols-2 gap-3 text-xs">
        <label v-for="s in sliders" :key="s.name" class="flex flex-col gap-1">
          <span class="flex justify-between text-fg-subtle">{{ s.name }} <span class="num text-fg">{{ s.model.value.toFixed(2) }}</span></span>
          <input v-model.number="s.model.value" type="range" :min="s.min" :max="s.max" step="0.01" class="accent-queen-400">
        </label>
      </div>
      <div class="flex flex-col gap-3 text-xs">
        <div class="rounded-lg border border-line bg-sunken p-3">
          <p class="text-fg-subtle">Gradient check (like <code>libs/autodiff/tests</code>)</p>
          <p class="num mt-1">
            engine ∂L/∂w <span class="text-queen-300">{{ graph.grads.w.toFixed(6) }}</span> · finite difference
            <span class="text-signal-300">{{ numericalDw.toFixed(6) }}</span>
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn-primary btn-sm" @click="step">Gradient step on w, b</button>
          <button class="btn-ghost btn-sm" @click="reset">Reset</button>
          <span class="num ml-auto text-fg-muted">L = {{ graph.L.toFixed(4) }}</span>
        </div>
        <div v-if="history.length" class="flex h-10 items-end gap-0.5">
          <div
            v-for="(l, i) in [...history, graph.L]"
            :key="i"
            class="flex-1 rounded-sm bg-queen-400/70"
            :style="{ height: `${Math.max(4, (l / Math.max(...history, graph.L, 1e-9)) * 100)}%` }"
          />
        </div>
      </div>
    </div>
    <figcaption class="mt-3 text-xs text-fg-subtle">
      <span class="text-signal-300">Blue</span>: forward value. <span class="text-queen-300">Pink</span>: gradient of L with
      respect to that node, filled in by the backward pass. Drag a slider and both passes re-run; press the button to move
      w and b against their gradients and watch the loss shrink.
    </figcaption>
  </figure>
</template>
