<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import type { GenerationStats, RunInfo } from "~/types/telemetry"

const linearRunCode = `def run(self, inputs: Sequence[float]) -> list[float]:
    registers = [0.0] * self.num_registers
    for instr in self.instructions:
        a = registers[instr.src_a % self.num_registers]
        b_idx = instr.src_b % (self.num_registers + self.num_inputs)
        b = registers[b_idx] if b_idx < self.num_registers else inputs[b_idx - self.num_registers]
        op = self.ops[instr.op % len(self.ops)]
        registers[instr.dst % self.num_registers] = op(a, b)
    return registers`

const treeCode = `@dataclass(frozen=True, slots=True)
class Terminal:
    value: float | None  # None means "the input variable x"; otherwise a random constant

    def evaluate(self, x, ops=DEFAULT_OPS):
        return x if self.value is None else self.value


@dataclass(frozen=True, slots=True)
class FunctionNode:
    op: int  # index into ops
    left: Node
    right: Node

    def evaluate(self, x, ops=DEFAULT_OPS):
        return ops[self.op % len(ops)](self.left.evaluate(x), self.right.evaluate(x))`

const crossoverCode = `def vary(self, parents, rng):
    a, b = parents[0], parents[1]
    # swap a random subtree of a for a random subtree of b...
    crossover_point = rng.randrange(len(_flatten(a.root)))
    donor = rng.choice(_flatten(b.root))
    child_root = _replace_at(a.root, crossover_point, donor)
    # ...then, sometimes, replace a random subtree with a fresh random one
    if rng.random() < self._mutation_rate:
        ...
    # bloat control: an offspring deeper than the limit is thrown away
    if _depth(child_root) > self._max_tree_depth:
        return a
    return replace(a, root=child_root)`

const paretoCode = `# linear GP: complexity = instructions that can reach the output register
ParetoSelection(complexity=lambda p: p.effective_instruction_count(), k=3)

# tree GP: complexity = nodes in the tree -- the same selection class, unchanged
ParetoSelection(complexity=lambda t: t.node_count(), k=3)`

// A real linear-GP champion from this project's own runs, shown with its introns marked.
const api = useApi()
const champion = ref<{ runId: string; ref: string } | null>(null)
onMounted(async () => {
  try {
    const runs = await api.fetch<RunInfo[]>("/runs")
    const run = runs.find((r) => r.config?.representation === "linear_gp" && r.status === "completed")
    if (!run) return
    const history = await api.fetch<GenerationStats[]>(`/runs/${run.run_id}/metrics/history`)
    const last = history.at(-1)
    if (last) champion.value = { runId: run.run_id, ref: last.champion_ref }
  } catch {
    // No backend: the chapter reads fine without the live example.
  }
})
</script>

<template>
  <article class="prose-chapter">
    <p>
      Chapter 1's loop -- evaluate, select, vary, repeat -- never says what a candidate solution
      <em>is</em>. That choice, the <strong>genome representation</strong>, decides what evolution can
      build, which mutations are even possible, and how easily good parts survive being recombined.
      This project implements three: a <strong>linear program</strong> (a tiny register machine), an
      <strong>expression tree</strong>, and -- in the Snake case study -- a neural network's
      <strong>weight vector</strong>.
    </p>

    <Callout variant="note" title="Two different “representations”">
      This chapter is about how a <em>solution</em> is encoded. How a <em>game</em> is shown to a
      solution -- raw grid vs. hand-engineered features -- is a separate choice, the observation
      representation, covered on the <NuxtLink to="/games/snake#leaderboard">Snake leaderboard</NuxtLink>
      and in docs/design/0007. Both matter; they're independent axes.
    </Callout>

    <h2>Linear GP: a tiny register machine</h2>
    <p>
      A <code>LinearProgram</code> is a fixed-length list of instructions. Each one reads two values
      (a register, or a register-or-input), applies an operator (+, −, ×, protected ÷), and writes the
      result to a register. Register <code>r0</code> after the last instruction is the output. This is
      the whole interpreter:
    </p>
    <CodeBlock lang="python" :code="linearRunCode" />
    <p>
      Every field is taken modulo its range, so <em>any</em> random integers form a valid program --
      mutation can change a single number anywhere and the result still runs. That's a big part of why
      linear GP is easy to evolve: there's no syntax to break.
    </p>

    <h2>Introns: code that doesn't matter</h2>
    <p>
      Because instructions write to shared registers, many of them end up computing values that are
      overwritten or never read on the way to <code>r0</code>. These are <strong>structural introns</strong>
      (the term comes from genetics: DNA that doesn't code for anything). A backward liveness pass finds
      them -- walk from the last instruction up, tracking which registers can still reach the output.
      Here's a real champion from one of this project's symbolic-regression runs, with its introns
      struck through:
    </p>
    <figure v-if="champion" class="card my-6 p-5">
      <ClientOnly><ChampionProgram :run-id="champion.runId" :champion-ref="champion.ref" /></ClientOnly>
      <figcaption class="mt-2 text-xs text-fg-subtle">
        Final champion of run <NuxtLink :to="`/runs/${champion.runId}`">{{ shortId(champion.runId) }}</NuxtLink>
        (evolving <code>f(x) = x²</code>), loaded live from the telemetry store.
      </figcaption>
    </figure>
    <p v-else class="text-sm text-fg-subtle">(Start the backend to see a real champion program here.)</p>
    <p>
      Introns aren't just waste. Mutations that land in them change nothing, so they act as a buffer:
      a population can drift through neutral changes and occasionally "switch on" an intron that turns
      out to be useful. They also make <em>size</em> a slippery measure -- a 12-instruction program might
      really be a 3-instruction one. That's why this project measures linear-GP complexity with
      <code>effective_instruction_count()</code>, not the raw length.
    </p>

    <h2>Tree GP: expressions as trees</h2>
    <p>
      The classic Koza-style representation: a program <em>is</em> an expression tree. Leaves are the
      input <code>x</code> or random constants; internal nodes apply an operator to their two children.
      Evaluation is plain recursion:
    </p>
    <CodeBlock lang="python" :code="treeCode" />
    <p>
      Variation works on whole subtrees: <strong>crossover</strong> swaps a random subtree of one parent
      for a random subtree of the other, and <strong>mutation</strong> replaces a random subtree with a
      fresh random one. Unlike a linear program, a tree's size isn't fixed -- and left alone, crossover
      makes trees grow without bound (<strong>bloat</strong>). The standard fix, used here, is a depth
      limit:
    </p>
    <CodeBlock lang="python" :code="crossoverCode" />

    <h2>Head to head</h2>
    <p>
      <code>notebooks/0003-linear-vs-tree.ipynb</code> ran both on the same benchmark
      (<code>x⁴ − 3x² + 2</code>, 11 sample points), same tournament selection (k=3), population 60,
      60 generations, 10 random seeds each. Fitness is negative error, so closer to 0 is better:
    </p>
    <div class="my-6 grid gap-3 sm:grid-cols-2">
      <div class="card p-4">
        <p class="text-[11px] tracking-wide text-fg-subtle uppercase">Linear GP (12 instructions, 4 registers)</p>
        <p class="num mt-1 text-2xl font-semibold text-fg">−0.284</p>
        <p class="text-xs text-fg-subtle">final mean best fitness · std 0.177 · started at −0.501</p>
      </div>
      <div class="card p-4">
        <p class="text-[11px] tracking-wide text-fg-subtle uppercase">Tree GP (max depth 4, up to 31 nodes)</p>
        <p class="num mt-1 text-2xl font-semibold text-life-300">−0.126</p>
        <p class="text-xs text-fg-subtle">final mean best fitness · std 0.152 · started at −0.485</p>
      </div>
    </div>
    <Callout variant="finding" title="Tree GP won this benchmark -- and that's all it shows">
      A plausible reason: a tree composes naturally (a subtree computing x² slots straight into a
      parent that squares it again), while a register machine has to <em>route</em> intermediate values
      through the right registers, which crossover easily disrupts. But the size budgets aren't
      perfectly comparable (there's no exact equivalence between 12 instructions and depth 4), and this
      is one smooth polynomial. A differently shaped problem could favor linear GP. It's a measured
      result, not a general law.
    </Callout>

    <h2>Same algorithms, any genome</h2>
    <p>
      The more important result from that notebook wasn't who won. Adding tree GP needed a new genome
      module and a new variation operator -- and nothing else. <code>evolve()</code>, tournament,
      lexicase, and Pareto selection were all reused unchanged, because they only ever look at fitness
      values, never inside a genome. Even Pareto selection's complexity objective is just a function you
      pass in:
    </p>
    <CodeBlock lang="python" :code="paretoCode" />
    <p>
      The third representation took that further: a neural network's weights, a genome that isn't a
      program at all. The same loop evolved it to play Snake -- that's the
      <NuxtLink to="/learn/teaching-a-snake">next chapter</NuxtLink>.
    </p>
  </article>
</template>
