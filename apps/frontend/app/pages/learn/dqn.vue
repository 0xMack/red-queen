<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.

const lossCode = `for i in 0..size {
    // the target: reward so far, plus the discounted value of the best next move -- read from the TARGET network
    let row = &next_target[i * n..(i + 1) * n];
    let chosen = match &next_online {
        Some(next) => argmax(&next[i * n..(i + 1) * n]),   // Double DQN: the online net picks the move...
        None => argmax(row),                               // ...plain DQN: the target net picks it too
    };
    let y = batch.returns[i] + batch.discounts[i] * row[chosen];
    let predicted = q[i * n + batch.actions[i]];
    let delta = y - predicted;                             // the same TD error the table used
    // Huber loss: quadratic near the target, linear far away -- a big surprise can't make a huge step
    loss += if delta.abs() <= 1.0 { 0.5 * delta * delta } else { delta.abs() - 0.5 };
    q_grad[i * n + batch.actions[i]] = -delta.clamp(-1.0, 1.0) / size as f64;
}
// then backpropagate q_grad through the network and take an Adam step
grads = online.backward(&cache, &q_grad);`

const stabilizersCode = `// experience replay: every transition goes into a ring buffer of the last 50,000...
replay.push(&stored);
// ...and every 4th step, the network trains on 32 of them drawn at random
let (indices, weights) = replay.sample(self.batch_size, beta, &mut self.sample_rng);

// a target network: the targets come from a frozen copy, refreshed every 2,000 steps
if self.target_update > 0 && self.observed.is_multiple_of(self.target_update) {
    self.target = Some(self.online.clone());
}`
</script>

<template>
  <article class="prose-chapter">
    <p>
      <NuxtLink to="/learn/q-learning">Q-learning</NuxtLink> kept one number per situation and move, and it stopped at the greedy baseline because
      its 11 features lump different situations into the same row. The obvious fix is to let the snake see more -- but every extra thing it sees
      multiplies the rows, and a table learns nothing about a row it hasn't visited. <strong>Deep Q-networks</strong> (DQN; Mnih et al., 2015,
      the method that learned Atari games from pixels) replace the table with a neural network, so what the agent learns in one situation carries
      over to situations that look like it.
    </p>

    <h2>From a table to a network</h2>
    <p>
      A table <em>looks up</em> Q(s, a). A Q-network <em>computes</em> it: the observation goes in, three numbers come out -- one value per move --
      and the agent picks the biggest, exactly as before. Here the network has two hidden layers of 64 ReLU units, about 6,000 weights for the
      27-number observation used below. The table for that observation couldn't even be written down: most of its inputs are distances, not yes/no.
    </p>
    <p>
      The price is that updates are no longer local. Changing a table cell changes one situation's value; a gradient step on a network nudges its
      estimate for <em>every</em> situation, because they all share the same weights. That is where the generalization comes from, and, as you'll
      see below, where the danger comes from too.
    </p>

    <h2>The update, as a loss</h2>
    <p>
      The learning rule is still the Bellman update: move Q(s, a) toward <em>r + γ · max Q(s′, ·)</em>. With a network there is no cell to move, so
      the difference becomes a <strong>loss</strong> -- how far the prediction is from its target -- and backpropagation (the
      <NuxtLink to="/learn/autodiff">autodiff chapter</NuxtLink>'s machinery, written out by hand in the Rust core) turns it into a gradient step on
      every weight. The target is treated as a constant: the network is pulled toward its own next-step estimate, not the other way round.
    </p>
    <CodeBlock lang="rust" :code="lossCode" />
    <p>
      This is the whole update, as the project's Rust core runs it. It is checked against an independent implementation on the project's own
      autodiff engine: same network, same minibatch, gradients equal to twelve decimal places.
    </p>

    <h2>Watch it learn</h2>
    <p>
      The same code, compiled to WebAssembly, training in your browser. It starts with random weights and trains for 200,000 moves; every 10,000
      moves the current network plays 30 games it never trained on. Train it once as it is (the snake sees along seven rays around its head, plus where the
      food and its own tail are), then switch <em>what the snake sees</em> to <code>features.v1</code> -- the Q-table's 11 yes/no features -- and
      train again: the first run stays on the chart for comparison.
    </p>
    <ClientOnly>
      <DqnLab />
      <template #fallback>
        <div class="card not-prose my-6 p-6 text-center text-sm text-fg-subtle">Loading the DQN lab…</div>
      </template>
    </ClientOnly>
    <p>
      On a desktop the first run passes the greedy heuristic within a few seconds and reaches the mid-20s by 200,000 moves. The
      <code>features.v1</code> run stops where the table stopped, around 18-19. The bars under the board are the three values the network is
      choosing between on the move being played.
    </p>

    <h2>Why it can blow up</h2>
    <p>
      Put three things together -- a function that generalizes, targets built from its own estimates, and learning about the best move while
      playing a different (exploring) one -- and the estimates can feed on themselves: an update that raises one value raises the target of the
      next, which raises the value again. Sutton and Barto call this the <strong>deadly triad</strong>, and a DQN has all three. DQN's answer is
      two changes to how the data is used, rather than to the rule itself:
    </p>
    <ul>
      <li>
        <strong>Experience replay.</strong> Instead of learning from each move as it happens -- consecutive moves are nearly the same situation --
        store the last 50,000 transitions and train on random handfuls of them. Updates stop chasing whatever the snake happens to be doing, and
        every transition gets used several times.
      </li>
      <li>
        <strong>A target network.</strong> Compute the targets with a frozen copy of the network, refreshed every 2,000 steps, so an update
        can't move the very target it's chasing.
      </li>
    </ul>
    <CodeBlock lang="rust" :code="stabilizersCode" />
    <p>
      Switch both off in the lab and train with seed 0: the values climb into the millions and the snake never learns to eat. Then try any other
      seed: all of the lab's other nine learn normally (24-27). The project ran each setup twenty times, and without a target network <strong>1 run
      in 20 diverged</strong> -- rarely, but completely. (Seed 0 is the unlucky one both in the experiment and in your browser, although the two
      play different games; what they share is the network's starting weights and its exploration draws.) These are the two runs from the
      experiment that diverged, next to a healthy one:
    </p>
    <DqnDivergence />

    <h2>What each fix buys</h2>
    <p>
      The original DQN was followed by a series of refinements, each aimed at a specific weakness. The project added them one at a time, each rung
      compared with the one above it:
    </p>
    <ul>
      <li><strong>Double DQN</strong> -- the online network picks the next move, the target network values it (the maximum of noisy estimates is biased upward).</li>
      <li><strong>Dueling heads</strong> -- one output for "how good is this situation", one for "how much better is each move", added back together.</li>
      <li><strong>3-step returns</strong> -- use three real rewards before bootstrapping, as the Q-learning chapter's n-step option did.</li>
      <li><strong>Prioritized replay</strong> -- replay the transitions the network got most wrong more often.</li>
    </ul>
    <DqnResults view="ladder" />
    <ul>
      <li>
        <strong>The target network is the one that matters here</strong>, and for stability rather than score: none of the 55 runs that had one
        diverged. Leaving out the two collapsed runs, replay and the target network are each worth under a point.
      </li>
      <li>
        <strong>Double DQN, dueling heads and prioritized replay changed nothing measurable.</strong> Double DQN barely lowered the average value
        estimate (2.59 against 2.65): with rewards of about ±1, a discount of 0.95 and only three moves, there is little upward bias to remove.
      </li>
      <li>
        <strong>3-step returns and more training helped a little, every time</strong> -- better in all five pairs of runs, about +0.9 each. Five
        million moves gave 30.5.
      </li>
    </ul>
    <Callout variant="finding" title="Found by running it: five seeds couldn't see the failure">
      The first run of this experiment had five seeds per rung, and one diverged in each of the two rungs without a target network -- which could
      mean a 20% failure rate or a fluke. Fifteen more seeds each made it about 5%. A rare failure needs more runs than an average does; when an
      arm's spread is one outlier, add seeds before drawing the conclusion.
    </Callout>
    <Callout variant="note" title="Dueling heads vanish on export">
      V(s) + A(s, a) − mean A is a linear function of the last hidden layer, so when a trained dueling network is saved, its two heads are folded
      into one ordinary output layer. Every DQN the project exports is a plain network, and it runs in the browser like any other model.
    </Callout>

    <h2>What the snake sees</h2>
    <p>
      The table's ceiling was aliasing: situations with different futures sharing one row. A network can generalize between rows, but it cannot
      separate situations that its input makes identical -- so the real question for this chapter was whether a gradient learner could make use of
      a <em>richer</em> observation. Evolution couldn't: NEAT did no better on <code>egocentric.v1</code> than on <code>features.v1</code>.
    </p>
    <DqnResults view="observers" />
    <ul>
      <li>
        <strong>On the Q-table's own features, a network gains nothing</strong> (18.8, the table's level). Generalizing doesn't help when the
        inputs can't tell the situations apart.
      </li>
      <li>
        <strong>On <code>egocentric.v1</code> it gains ten points.</strong> The rays say how far away the nearest wall, body segment and piece of
        food are in seven directions, which separates many of the situations the 11 features merge.
      </li>
      <li>
        <strong><code>grid-flat.v1</code> fails for every learner so far.</strong> It lists all 100 cells as numbers 0-3 (empty, body, head, food) in
        the board's frame, while the moves are turns relative to the head: the categories share one number line, and the heading is only implicit.
        Evolution scored 0.06-0.17 on it; the DQN 0.38.
      </li>
    </ul>
    <p>
      The best DQN is on the <NuxtLink to="/games/snake">Snake leaderboard</NuxtLink> at <strong>29.3, second</strong> behind NEAT's 38 -- trained
      in 6.6 minutes on five million moves, where NEAT's champion took 409 million. It plays in your browser from the same kind of model package
      as every other entrant.
    </p>

    <h2>Where this goes next</h2>
    <p>
      Two directions. The observation is the lever that has moved Snake scores for both evolution and value learning, and the analysis of NEAT's
      deaths already names what's missing: rays can't tell whether the space ahead is enclosed. (Tested since: <code>egocentric.v2</code>
      adds, for each move, how much of the board stays reachable, and it lifts this same DQN from 28.5 to 41.5 -- see the
      <NuxtLink to="/learn/policy-gradients">policy-gradients chapter</NuxtLink>.) And every method so far learns <em>values</em> and
      acts on them; <strong>policy gradients</strong> learn the policy directly -- the probabilities of each move -- which avoids the maximum at the
      heart of the deadly triad, and leads on to the actor-critic methods (A2C, PPO) behind most of modern reinforcement learning.
    </p>
  </article>
</template>
