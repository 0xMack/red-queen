<script setup lang="ts">
import type { RenderState } from "~/types/games"

// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.

const discretizerCode = `// Snake's features.v1: 11 numbers, each 0 or 1 -> one of 2^11 = 2,048 rows
Discretizer::Binary { bits } => observation
    .iter()
    .enumerate()
    .map(|(i, &v)| if v != 0.0 { 1 << i } else { 0 })
    .sum(),`

const updateCode = `/// Move the oldest pending transition towards its n-step return ending in \`bootstrap\`.
fn update_oldest(&mut self, bootstrap: f64) {
    let mut g = bootstrap;
    for &(_, _, reward) in self.pending.iter().rev() {
        g = reward + self.gamma * g;              // r + γ·(what comes after)
    }
    let (state, action, _) = self.pending.pop_front().expect("something pending");
    let cell = state * self.actions + action;
    let td = g - self.q[cell];                   // how wrong the table was
    self.q[cell] += self.alpha * td;             // ...move a fraction α of the way
}`

const learnCode = `pub fn learn(&mut self, step: Step) {
    self.pending.push_back((step.state, step.action, step.reward));
    let value_next = |agent: &Self| match (agent.sarsa, step.next_action) {
        (true, Some(a)) => agent.row(step.next_state)[a],        // SARSA: the move it will make
        _ => {
            let row = agent.row(step.next_state);
            row[agent.greedy(step.next_state)]                  // Q-learning: the best move
        }
    };
    if step.done {
        self.flush(0.0);                          // the game ended: nothing comes after
    } else if step.truncated {
        let bootstrap = value_next(self);
        self.flush(bootstrap);                    // cut off by the step cap: the game would have gone on
    } else if self.pending.len() == self.n_step {
        let bootstrap = value_next(self);
        self.update_oldest(bootstrap);
    }
}`

const epsilonCode = `fn epsilon_greedy(&self, state: usize, rng: &mut Rng) -> usize {
    if rng.uniform() < self.epsilon() {
        rng.below(self.actions as u32) as usize   // explore: any move, at random
    } else {
        self.greedy(state)                       // exploit: the best move the table knows
    }
}

// ε falls in a straight line from epsilon_start (1.0) to epsilon_end (0.05)
// over the first epsilon_decay_steps (100k) moves, then stays there.`

// Two boards the table can't tell apart (docs/design/0010, "Why it stops"). Both: heading right, head at (5,5),
// food up and to the right, nothing in the three cells around the head -- the same features.v1 row. On the left,
// turning up is a fine move. On the right, the cell above the head is a pocket walled in by the snake's own body,
// and turning up is certain death two moves later. Cells are ordered like render_state(): body nearest the head
// first, the head, then food.
function board(body: [number, number][]): RenderState {
  return {
    width: 10,
    height: 10,
    score: 0,
    alive: true,
    cells: [
      ...body.map(([x, y]) => ({ x, y, label: "body" })),
      { x: 5, y: 5, label: "head" },
      { x: 8, y: 2, label: "food" },
    ],
  }
}
const openBoard = board([
  [4, 5],
  [3, 5],
])
const pocketBoard = board([
  [4, 5],
  [4, 4],
  [4, 3],
  [5, 3],
  [6, 3],
  [6, 4],
  [7, 4],
  [7, 5],
])
</script>

<template>
  <article class="prose-chapter">
    <p>
      Every chapter so far has trained a player the same way: make a population of candidates, let each one play whole games, keep the ones with the
      best scores. Evolution only ever sees a game's <em>final score</em>. It never learns which moves in that game were good -- a snake that ate
      twelve pieces of food and then drove into a wall gets one number, and so does its offspring.
    </p>
    <p>
      <strong>Reinforcement learning</strong> starts from the opposite end. There is one agent, not a population, and it learns from every single
      move: it acts, the game says what happened (a <strong>reward</strong>), and it adjusts. This chapter builds the oldest and simplest version,
      <strong>Q-learning</strong> (Watkins, 1989), runs it on the same Snake the evolved networks play, and measures where it gets to -- and
      where it stops.
    </p>

    <h2>Learning from experience</h2>
    <p>
      The setup has four parts. At each step the agent sees an <strong>observation</strong> (for Snake: the same 11 features the evolved networks
      read -- danger ahead, left and right; which way it's heading; which way the food is), picks an <strong>action</strong> (turn left, go
      straight, turn right), and gets back a <strong>reward</strong> and the next observation. Snake's reward is what the game already reports:
      <strong>+1</strong> for eating, <strong>−1</strong> for dying, and a small nudge otherwise (+0.01 for moving closer to the food, −0.02 for moving
      away).
    </p>
    <p>
      The agent doesn't want the biggest reward <em>now</em>; it wants the biggest total from here on. That total, with later rewards counted a
      little less than sooner ones, is the <strong>return</strong>:
    </p>
    <p class="text-center font-mono text-sm">G = r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + …</p>
    <p>
      γ (gamma, the <strong>discount</strong>) is a number just below 1. At 0.95, a reward twenty moves away counts for about a third of one
      right now. It keeps the sum finite and expresses a sensible preference: food now beats the same food later.
    </p>

    <h2>A value for every move in every situation</h2>
    <p>
      Q-learning keeps one number for each pair of situation and move: <strong>Q(s, a)</strong>, its current estimate of the return it will get if
      it makes move <em>a</em> in situation <em>s</em> and plays well afterwards. If those numbers were right, playing would be trivial -- in each
      situation, pick the move with the biggest Q.
    </p>
    <p>
      With 11 yes/no features there are 2<sup>11</sup> = 2,048 possible situations, so the whole agent is a <strong>table</strong>: 2,048 rows,
      three columns, 6,144 numbers. A situation's row number is just its features read as a binary number:
    </p>
    <CodeBlock lang="rust" :code="discretizerCode" />
    <p>
      Only 256 of those rows can ever happen (the snake always heads exactly one way, and the food is never in two opposite directions at once),
      and those are the ones drawn in the grid under the live demo below.
    </p>

    <h2>The update rule</h2>
    <p>
      The table starts at zero. What makes it learnable is that the value of a move can be written in terms of the value of the <em>next</em>
      situation: the return from here is the reward for this move, plus γ times the return from wherever it lands. That's the
      <strong>Bellman equation</strong>, and Q-learning turns it into an update. After each move, compare what the table predicted with what one
      step of real experience suggests, and move the prediction part of the way there:
    </p>
    <p class="text-center font-mono text-sm">Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s′, ·) − Q(s, a) ]</p>
    <p>
      The bracket is the <strong>temporal-difference error</strong> -- how surprised the table was. α (alpha, the learning rate) is how far to move:
      0.1 means a tenth of the way, so the value becomes an average over many visits rather than whatever happened last time. Here is the update,
      as the project's Rust core does it:
    </p>
    <CodeBlock lang="rust" :code="updateCode" />
    <p>
      Notice what this does <em>not</em> need: a finished game. The table learns from each step using its own guess about the next situation
      (<strong>bootstrapping</strong>). One piece of food propagates backwards over many games -- first to the move that ate it, then to the move
      before that, and so on. The code above is slightly more general than the formula. It can wait <em>n</em> steps and use n real rewards before
      bootstrapping (<strong>n-step returns</strong>), and it handles the end of a game explicitly:
    </p>
    <CodeBlock lang="rust" :code="learnCode" />
    <p>
      There are two ways a game can stop. When the snake dies, there is no future, so nothing is added after the last reward. When a game is merely
      <em>cut off</em> by a step limit, the snake would have gone on playing, so the value of where it stopped still counts. Treating those alike
      is a classic bug: the agent learns that surviving a long time is as bad as dying.
    </p>
    <Callout variant="note" title="Q-learning and SARSA differ in one line">
      Q-learning bootstraps from the <em>best</em> next move (max Q), whatever it actually does next -- it learns the value of playing perfectly
      (<em>off-policy</em>). <strong>SARSA</strong> (state, action, reward, state, action) bootstraps from the move it's actually about to make,
      exploration included -- it learns the value of the way it really plays (<em>on-policy</em>), which makes it more cautious near danger while it
      is still exploring. In the code it's the <code>match</code> at the top of <code>learn</code>.
    </Callout>

    <h2>Exploring on purpose</h2>
    <p>
      A table that always picks its current best move gets stuck. Early on every value is zero, the first move that happens to earn a little becomes
      "best", and the moves it never tries never get the chance to look better. So the agent sometimes picks a move at random:
      <strong>ε-greedy</strong> exploration.
    </p>
    <CodeBlock lang="rust" :code="epsilonCode" />
    <p>
      There's another way to explore, with no randomness at all: start every value <em>high</em> (<strong>optimistic initialization</strong>).
      Every untried move looks better than anything tried so far, so the greedy agent tries each of them until experience drags its value down to
      reality. The demo's "optimistic start" does this, with Q starting at 2 and ε held at a token 0.02.
    </p>

    <h2>Watch it learn</h2>
    <p>
      This is that same Rust code, compiled to WebAssembly and running in your browser in a background thread. Press <strong>Train</strong>: the
      table starts empty and trains for a million moves. Every 25,000 moves it plays 30 games it never trains on with no exploration, and the
      average score goes on the chart. The board shows the current table playing greedily, move by move, with the row it's in and the three values it
      is choosing between. The grid underneath is the whole reachable table, one cell per situation, colored by the move it currently prefers.
    </p>
    <ClientOnly>
      <QLearningLab />
      <template #fallback>
        <div class="card not-prose my-6 p-6 text-center text-sm text-fg-subtle">Loading the Q-learning lab…</div>
      </template>
    </ClientOnly>
    <p>Things to watch for:</p>
    <ul>
        <li>
          The grid fills in unevenly. Random early moves reach about a hundred situations in the first 5,000 moves, but a snake moving at random
          stays short, and short snakes rarely have their own body on two sides of the head. Those rows only fill in once the table is good enough
          to grow a long snake, around 100,000 moves in. What an agent can learn about depends on where its own play takes it.
        </li>
        <li>
          The curve rises fast, and then it goes flat. By about 150,000 moves it reaches the greedy heuristic's line, and from there it
          wanders a point or two either side of it (30 games is a noisy measurement). Training longer doesn't move it.
        </li>
        <li>
          One rule it clearly learns is the obvious one: in the half of the table with danger ahead, the cells go blue and amber. The
          recorded run's final table goes straight into danger in only 4 of those 128 rows.
        </li>
    </ul>

    <h2>What the knobs do</h2>
    <p>
      One run is an anecdote -- a different random seed moves a Snake score by a point or two -- so the project ran each setting five times, a
      million moves each (<code>jobs/rl_experiment.py</code>), and scored every final table on the same 200 held-out games the leaderboard uses:
    </p>
    <QLearningResults />
    <ul>
      <li>
        <strong>Nothing is significantly better than the defaults.</strong> The best setting, optimistic start, averaged 19.4 against 17.6 (p = 0.19).
        The spread between seeds of the same setting is about as large as the spread between settings.
      </li>
      <li>
        <strong>Five times the training changes nothing</strong> (17.9 after 5M moves). Every run, of every setting, had visited all 256 reachable
        rows. The table has converged, and extra data can't fix it.
      </li>
      <li>
        <strong>It's cheap.</strong> A million moves takes about half a second in the Rust core. For comparison, the best evolved champion took
        409 million moves and 41 minutes.
      </li>
    </ul>
    <p>
      The best of these tables is on the <NuxtLink to="/games/snake">Snake leaderboard</NuxtLink> -- exported as a model package, like every
      other champion, and played in your browser -- at 19.5, just above the greedy heuristic.
    </p>

    <h2>Why it stops at the greedy baseline</h2>
    <p>
      Here is the puzzle. Any rule that looks only at these 11 features and picks a move <em>is</em> a table with 256 rows -- the evolved NEAT
      network that scores <strong>38</strong> reads exactly the same 11 features, so a 38-point table exists. The table has room for it; Q-learning
      just can't find it.
    </p>
    <p>
      The reason is that the features throw information away. They say whether the three cells next to the head are free, and nothing else about
      the body. So situations that are really very different end up in the same row:
    </p>
    <figure class="not-prose my-6 grid gap-4 sm:grid-cols-2">
      <div>
        <div class="mx-auto max-w-[240px]"><GridBoard :state="openBoard" :tick-ms="0" /></div>
        <figcaption class="mt-2 text-center text-xs text-fg-subtle">Turning up (left) heads for the food. A good move.</figcaption>
      </div>
      <div>
        <div class="mx-auto max-w-[240px]"><GridBoard :state="pocketBoard" :tick-ms="0" /></div>
        <figcaption class="mt-2 text-center text-xs text-fg-subtle">
          Turning up enters a pocket walled in by the snake's own body. Dead in two moves.
        </figcaption>
      </div>
      <p class="text-center text-sm text-fg-muted sm:col-span-2">
        Both are the same row: heading right, no danger next to the head, food up and to the right.
      </p>
    </figure>
    <p>
      A value learned for that row is an average over every situation that lands in it, the good ones and the deadly ones mixed together, and the
      move that looks best on average isn't the one that works best across the situations the snake actually meets. Worse, those averaged values
      are what every <em>other</em> row bootstraps from, so the error spreads. The math behind the update rule assumes the observation tells you
      everything that matters about the future (the <strong>Markov</strong> property), and here it doesn't -- a situation is called
      <strong>aliased</strong> when it shares a row with situations that have different futures.
    </p>
    <p>
      Evolution doesn't care. It never estimates values -- it just tries whole policies and keeps the ones that score -- so it can find a table that
      happens to work well on average across all the aliased situations. Learning from every step is far more efficient, but only when the
      steps are described well enough to learn from.
    </p>
    <Callout variant="finding" title="Found by running it: more data, same score">
      Before this experiment, the expected weak points were exploration and training length. Neither mattered: slower or faster ε decay, optimistic
      starts and 5× the training all landed within noise of each other, every run visited every reachable row, and the tables converged. The
      ceiling was the observation -- which is also why the <NuxtLink to="/learn/teaching-a-snake">Snake case study</NuxtLink> found that
      evolution's ceiling moved when its representation changed, not when its budget did.
    </Callout>

    <h2>Where this goes next</h2>
    <p>
      The fix is to give the learner an observation that says more -- rays that see the body further away, or the whole board -- and those don't
      fit in a table: there are far too many possible boards to give each one its own row, and the agent would never visit most of them. So the next step replaces the table with a
      neural network that <em>estimates</em> Q from the observation, so that similar situations share what they learn: <strong>deep
      Q-networks</strong> (DQN), the method that learned to play Atari games from pixels. It needs its own tricks -- a replay buffer, a slowly
      updated target network -- and its own chapter. After that comes learning the policy directly instead of values (<strong>policy
      gradients</strong>), and then two-player games learned by playing against itself.
    </p>
  </article>
</template>
