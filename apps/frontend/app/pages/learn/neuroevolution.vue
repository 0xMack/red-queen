<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
const forwardCode = `def _forward(weights, layer_sizes, observation):
    """A tanh-activated feedforward pass over one flat list of weights."""
    activations = list(observation)
    offset = 0
    for i in range(len(layer_sizes) - 1):
        in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
        next_activations = []
        for o in range(out_size):
            total = weights[offset + in_size * out_size + o]  # this unit's bias
            for k in range(in_size):
                total += weights[offset + o * in_size + k] * activations[k]
            next_activations.append(math.tanh(total))
        offset += in_size * out_size + out_size
        activations = next_activations
    return activations`

const genomeCode = `@dataclass(frozen=True, slots=True)
class WeightVector:
    weights: tuple[float, ...]     # every weight and bias, flattened
    layer_sizes: tuple[int, ...]   # e.g. (11, 16, 3) -- fixed for the whole run`

const fitnessCode = `def act(genome, observation):
    outputs = genome.forward(observation)        # 11 features in, 3 numbers out
    return interface.action.decode(outputs)      # argmax -> turn left / straight / right

fitness = SimulationFitnessEvaluator(envs=games, act=act, max_steps=200)
fitness.evaluate(genome)   # -> one total reward per game: the genome's "test cases"`

const mutationCode = `class GaussianMutation:
    def __init__(self, sigma: float = 0.1):
        self._sigma = sigma

    def vary(self, parents, rng):
        parent = parents[0]
        new_weights = tuple(w + rng.gauss(0.0, self._sigma) for w in parent.weights)
        return replace(parent, weights=new_weights)`

const loopCode = `evolve(
    population,                                  # 100 random WeightVectors
    fitness=fitness,
    selection=LexicaseSelection(),               # or TournamentSelection -- any strategy works
    variation=GaussianMutation(sigma=0.2),       # the only genetic operator
    generations=250,
)`
</script>

<template>
  <article class="prose-chapter">
    <p>
      <NuxtLink to="/learn/genetic-algorithms">Chapter 1</NuxtLink>'s loop -- evaluate, select, vary -- works on any genome, and the previous
      chapter gave it two program-shaped ones. <strong>Neuroevolution</strong> gives it a neural network instead. There is no gradient and no
      backpropagation: the network is treated as a black box that maps observations to actions, scored by how well it plays, and improved by
      mutating copies of the ones that did best. It's the technique behind the Snake policy in
      <NuxtLink to="/learn/teaching-a-snake">the Snake case study</NuxtLink>; this chapter opens it up.
    </p>

    <h2>A network is a list of numbers</h2>
    <p>
      The first decision is what a genome <em>is</em>. Here it is deliberately plain: the network's <strong>shape is fixed</strong> in advance
      (say, 11 inputs → 16 hidden units → 3 outputs) and the genome is every weight and bias, flattened into one list, layer by layer. That
      is the whole data structure:
    </p>
    <CodeBlock lang="python" :code="genomeCode" />
    <p>
      Running it is the ordinary feedforward pass, reading numbers out of the list by position:
    </p>
    <CodeBlock lang="python" :code="forwardCode" />
    <p>
      Below is a tiny 2-3-1 network, so all thirteen numbers fit on screen. Click a cell to see which connection it is, drag it, and watch what
      happens to the XOR truth table. XOR is the classic test because no network without a hidden layer can compute it -- the best a
      layerless network can do is 0.75.
    </p>
    <WeightVectorExplorer />

    <h2>Fitness comes from playing</h2>
    <p>
      Supervised learning would compare the network's output to a labelled answer. Games don't come with labels: nobody can say what the
      right move is at a given moment, only how the whole game went. So a genome is scored by <em>playing</em> -- the network reads the
      board, picks a move, the game responds, repeat -- and the total reward is its fitness:
    </p>
    <CodeBlock lang="python" :code="fitnessCode" />
    <p>
      That one <code>evaluate()</code> call returns one number <em>per game</em>, not one overall, which is what lets
      <NuxtLink to="/learn/selection-strategies">lexicase selection</NuxtLink> treat each game as its own test case. Nothing about the network
      needs to be differentiable, which is why the same loop also works for genomes whose behaviour has hard edges, like an argmax over
      three moves.
    </p>

    <h2>Mutation is the whole search operator</h2>
    <p>
      To make a child, copy the parent and nudge <em>every</em> weight by a small random amount drawn from a bell curve of width
      <strong>σ</strong>. That's all of it:
    </p>
    <CodeBlock lang="python" :code="mutationCode" />
    <p>
      σ is the one knob that matters. Whether a mutation is useful depends on how big it is relative to how much the current network can
      tolerate. The plot below mutates one nearly-solved XOR network 400 times and compares each child's fitness to its parent's:
    </p>
    <MutationMicroscope />
    <Callout variant="note" title="This is Evolution Strategies">
      Keep the top few, add Gaussian noise, repeat -- that is an Evolution Strategy, decades older than deep RL. The connection to gradient
      methods is real: weight each noise direction by how much it helped and you get a stochastic estimate of the gradient of (smoothed)
      fitness -- the view behind natural evolution strategies and OpenAI's ES. Truncation selection, as used here, is a cruder version of the
      same idea. So neuroevolution and policy gradients are cousins: one estimates the gradient by sampling perturbations, the other computes
      it exactly through the network. <code>docs/design/0003</code> calls this "the case where RL and EA are the same algorithm".
    </Callout>

    <h2>The whole loop, live</h2>
    <p>
      Put the pieces together and this is the entire algorithm: score a population, keep the best fifth, refill with mutated copies, keep the
      champion untouched. Below it runs on XOR with a 2-3-1 network. Each column of dots is one generation; the green dot is its best. Try a
      tiny σ (progress crawls -- every child is nearly its parent), then a huge one (the population keeps overshooting), and see how
      small a population can still get there.
    </p>
    <NeuroEvoLab />
    <p>
      On Snake the code is the same shape, only bigger -- the identical <code>evolve()</code> from Chapter 1, with a
      <code>WeightVector</code> genome:
    </p>
    <CodeBlock lang="python" :code="loopCode" />

    <h2>Why there's no crossover</h2>
    <p>
      Chapter 3's genomes recombine: linear GP swaps instruction runs, tree GP swaps subtrees. Weight vectors here are <em>only</em> mutated,
      and the reason is worth understanding because it's the problem the next chapter is built to solve. Two networks can compute exactly the
      same function while storing completely different lists of numbers -- swap which hidden unit is "unit 0" and "unit 1" and nothing about
      the behaviour changes, but every weight moves. Breed those two by position and the child mixes half of one convention with half of
      another:
    </p>
    <PermutationDemo />
    <p>
      This is the <strong>competing conventions</strong> (or permutation) problem. It's why a plain weight-vector GA gets little from
      crossover and Evolution Strategies drop it entirely: the operator has no way to know which gene in one parent corresponds to which gene
      in the other.
    </p>

    <h2>What a fixed shape costs</h2>
    <p>
      The design is simple and it works -- the Snake champion came from exactly this loop. But look at what was decided <em>before evolution
      started</em>: how many hidden units (16, chosen by hand), how many layers, that every input connects to every hidden unit. Too small and
      the network can't represent the solution; too big and you're searching 243 dimensions when a handful might do, with every weight a place
      for noise to hide. Nothing in the loop can change its mind. Evolving the <em>structure</em> along with the weights is the natural next
      step, and it runs straight into the problem above -- which is where
      <NuxtLink to="/learn/neat">NEAT</NuxtLink> comes in.
    </p>
  </article>
</template>
