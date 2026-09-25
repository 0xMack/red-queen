<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.

const reinforceCode = `// log π(a | s) and its gradient w.r.t. the network's outputs (the logits of a softmax)
let logp = log_softmax(row);
let log_p = logp[a];                               // a: the move that was played
let d_log_p: Vec<f64> = (0..outputs).map(|j| (j == a) as u8 as f64 - p[j]).collect();

// REINFORCE / A2C: push log π(a | s) up in proportion to the advantage of what followed
loss -= advantage * log_p * scale;
// ...plus a small entropy bonus, so the policy doesn't commit before it has explored
loss -= entropy_coef * entropy * scale;`

const gaeCode = `/// GAE(λ): δ_t = r_t + γ V(s_t+1) − V(s_t),  A_t = δ_t + γλ A_t+1, restarting at every episode boundary.
for t in (0..n).rev() {
    let bootstrap = if done[t] { 0.0 } else { next_values[t] };   // a game that ended has no future
    let delta = rewards[t] + gamma * bootstrap - values[t];
    if cut[t] {
        next_advantage = 0.0;                                     // a new episode starts after t
    }
    next_advantage = delta + gamma * lambda * next_advantage;
    advantages[t] = next_advantage;
}
// λ = 1 and V = 0: plain Monte-Carlo returns (REINFORCE). λ = 1 and a learned V: returns minus a baseline.`

const ppoCode = `let ratio = exp(log_p - old_log_prob);             // how much more likely the move is than when it was played
let clamped = ratio.clamp(1.0 - eps, 1.0 + eps);   // eps = 0.2
let (unclipped, capped) = (ratio * advantage, clamped * advantage);
loss -= unclipped.min(capped) * scale;             // the pessimistic of the two
// the gradient flows only through the unclipped term, and only when it is the smaller one:
// once a move is 20% more (or less) likely than it was, this sample stops pushing it further
let weight = if unclipped <= capped { ratio * advantage } else { 0.0 };`
</script>

<template>
  <article class="prose-chapter">
    <p>
      <NuxtLink to="/learn/q-learning">Q-learning</NuxtLink> and <NuxtLink to="/learn/dqn">DQN</NuxtLink> learn how good each move is and then
      act on the best one. <strong>Policy-gradient</strong> methods skip the middle step: the network outputs the policy itself -- a probability
      for each move -- and training nudges those probabilities toward whatever turned out well. This chapter builds the family from the simplest
      version (REINFORCE, 1992) up to <strong>PPO</strong> (2017), the workhorse of modern reinforcement learning, and it turns out to be the
      strongest Snake player in the project by a wide margin.
    </p>

    <h2>Learning the policy itself</h2>
    <p>
      The policy network looks like the DQN's -- observation in, three numbers out -- but the three numbers are turned into probabilities (a
      <em>softmax</em>), and the snake <em>samples</em> its move from them while it trains. Exploration is no longer bolted on with ε: a policy
      that isn't sure spreads its probability, and one that is concentrates it. When it's judged on unseen games it plays its most likely move.
    </p>
    <p>
      Learning probabilities directly has two practical advantages. The policy can be genuinely random where randomness helps, and the output
      doesn't have to be one of a few discrete choices -- the <a href="#continuous-actions">continuous-actions</a> section below learns an
      acceleration, which no table of values can represent without chopping it into bins.
    </p>

    <h2>The policy-gradient theorem</h2>
    <p>
      We want to raise the expected return. The <strong>policy-gradient theorem</strong> says its gradient is an average over the moves the policy
      actually made: for each one, the gradient of the log-probability of that move, times how good things turned out afterwards.
    </p>
    <p class="text-center font-mono text-sm">∇ J = E[ G<sub>t</sub> · ∇ log π(a<sub>t</sub> | s<sub>t</sub>) ]</p>
    <p>
      Nothing in it needs a model of the game, or even a derivative of the game: the snake plays, and every move is made more likely if it was
      followed by a high return and less likely if not. <strong>REINFORCE</strong> is exactly that: play whole episodes, compute each move's
      return G<sub>t</sub> (the discounted rewards from there to the end), and step along that estimate.
    </p>
    <CodeBlock lang="rust" :code="reinforceCode" />
    <p>
      For a softmax the gradient of log π(a) with respect to the logits is simply <em>1 for the chosen move, minus the probabilities</em> -- the
      snippet's <code>d_log_p</code>. As everywhere in this project, the Rust is checked against the project's own autodiff engine: loss and gradients
      agree to twelve decimal places, for every variant in this chapter.
    </p>

    <h2>Baselines: the same gradient, less noise</h2>
    <p>
      REINFORCE's estimate is right on average and very noisy. In Snake every move in a long game is followed by a large return, so every move gets
      pushed up, the good ones only slightly more than the bad; telling them apart takes a great many games. The fix is to subtract a
      <strong>baseline</strong> from the return -- how good things usually are from this situation, V(s), learned by a second network (the
      <em>critic</em>). A move then gets pushed up only if it did <em>better than expected</em>. Subtracting anything that doesn't depend on the move
      leaves the average gradient unchanged; it only removes noise. What's left, G<sub>t</sub> − V(s<sub>t</sub>), is called the
      <strong>advantage</strong>.
    </p>

    <h2>Actor-critic</h2>
    <p>
      Once there is a critic, the policy doesn't have to wait for the end of the episode: after a few steps it can use the critic's estimate of the
      rest, the way Q-learning bootstrapped. That is <strong>actor-critic</strong> (the policy is the actor), and <strong>A2C</strong> updates every
      128 steps. The standard way to blend the two extremes -- real returns (unbiased, noisy) and the critic's guess (smooth, biased while it is still
      learning) -- is <strong>generalized advantage estimation</strong>, with λ sliding between them. The project uses one function for all three
      algorithms:
    </p>
    <CodeBlock lang="rust" :code="gaeCode" />

    <h2>Why PPO clips</h2>
    <p>
      Every method so far throws a batch of experience away after a single gradient step, because once the policy changes, the experience no
      longer describes what it would do. <strong>PPO</strong> reuses each batch for several epochs anyway, and keeps that safe by capping how far
      one batch can move the policy: it weighs each move by the <em>ratio</em> of its probability now to when it was played, and clips the ratio's
      effect to ±20%.
    </p>
    <CodeBlock lang="rust" :code="ppoCode" />
    <p>Here is what each idea bought, one rung at a time, on the observation DQN reached 28.5 with:</p>
    <PgResults view="ladder" />
    <ul>
      <li>
        <strong>Plain REINFORCE already matches DQN</strong> (28.6 against 28.5 -- though with twice the game steps), with no replay buffer, no
        target network and no ε.
      </li>
      <li><strong>The baseline is worth six points</strong> -- better in all five pairs of runs.</li>
      <li>
        <strong>A2C's bootstrapping didn't help here</strong> (−1.6, within noise). Its advantages lean on a critic that is still learning --
        less noise, but a bias -- and on this problem the trade came out even.
      </li>
      <li>
        <strong>PPO adds ten points</strong>, the largest single step on the ladder, and every run of it beats the best evolved champion (NEAT, 38).
      </li>
    </ul>

    <h2>Watch it learn</h2>
    <p>
      PPO, in your browser. The bars under the board are the policy's own probabilities for the three moves in the position being played; the lower
      chart is its <strong>entropy</strong>, how undecided it is on average, which starts at ln 3 (three equal choices) and falls as it commits. Try
      the other rungs, or the other observations, and compare.
    </p>
    <ClientOnly>
      <PolicyGradientLab />
      <template #fallback>
        <div class="card not-prose my-6 p-6 text-center text-sm text-fg-subtle">Loading the policy-gradient lab…</div>
      </template>
    </ClientOnly>
    <p>
      On a desktop it trains at about 6,500 moves a second: with <code>egocentric.v2</code> the default run passes the greedy heuristic almost at
      once, reaches the mid-40s by 200,000 moves and the 50s by 400,000.
    </p>

    <h2 id="continuous-actions">Continuous actions</h2>
    <p>
      Snake has three moves. Many problems -- steering, throttle, a robot's joints -- have a number to choose instead. A policy network handles that
      without any change to the method: it outputs the <em>mean</em> of a normal distribution, with a learned spread, and the agent samples its
      action from it. The gradient of the log-density takes the place of the softmax's, and everything else -- baselines, advantages, PPO's clipping --
      is the same code.
    </p>
    <p>
      Reach1D is the smallest such problem in the project: an agent on a line chooses an acceleration between −1 and 1 each step, and loses its
      distance to a target every step. Watch the policy's distribution (the bell curve) and the agent's route.
    </p>
    <ClientOnly>
      <ReachLab />
    </ClientOnly>
    <p>
      The mean soon learns to steer toward any target and brake as it arrives; the rest of training makes the approach faster (the return climbs
      from about −90 to −45 over 200,000 steps). Watch the spread: it first <em>widens</em> -- while the mean is still wrong, sampling further from
      it is what finds better actions -- then narrows to about a third of where it started once the mean is right. Nobody told it to; the gradient
      on the spread does that by itself. (The Snake lab adds a small entropy bonus to keep its policy from committing too early; here it's off, or
      the spread never visibly narrows.) A DQN can't do this at all -- it needs a value for every action, and there are infinitely many -- and the Q-table only
      managed it by cutting the acceleration into bins.
    </p>

    <h2>Gradients against evolution, on the same network</h2>
    <p>
      The <NuxtLink to="/learn/neuroevolution">neuroevolution chapter</NuxtLink> evolved an 11 → 16 → 3 tanh network on the 11 features: 243
      weights, found by mutation and selection. Train the <em>same</em> network on the <em>same</em> observation with PPO instead (its outputs skip
      the final tanh, which doesn't change which move is largest) and the two paradigms can be compared directly, seed for seed:
    </p>
    <PgResults view="evolution" />
    <p>
      <strong>Gradients won on every seed</strong>, by 3.2 points over neuroevolution's better selection method -- using an eighth of the game
      steps (2 million against 16.5 million) and about a twenty-fifth of the time (15 seconds against 6.4 minutes). Evolution judges a whole network
      by one number per game; the policy gradient gets a signal from every move of every game. That's the sample-efficiency argument this project set
      out to measure, now measured on identical networks.
    </p>
    <Callout variant="warning" title="What this doesn't show">
      On these 11 features, neither is a good player: both sit near the greedy heuristic, because the features themselves are the ceiling (see the
      Q-learning chapter). NEAT, which grows its own network, reached 38 on them with 409 million steps. The comparison is about how efficiently the
      same weights get tuned, not about which paradigm plays Snake best.
    </Callout>

    <h2>What the snake sees, again</h2>
    <p>
      The DQN chapter ended on the observation: rays that see along lines, but not whether the space ahead is enclosed. <code>egocentric.v2</code>
      adds exactly that -- for each move, the share of the board still reachable after it, and whether the snake's own tail is -- and it lifted a
      DQN from 28.5 to 41.5. It lifts PPO further:
    </p>
    <PgResults view="observers" />
    <p>
      <strong>63 points</strong>, every run between 59 and 66: two thirds more than the best evolved champion, from two million game steps and
      about three and a half minutes of training. Five times the training added about three more (66), though not on every seed -- the spread
      between runs grew as well.
    </p>

    <h2>Where this goes next</h2>
    <p>
      Every game so far has been a single player against the rules. <strong>Checkers</strong> has an opponent, and the strongest way to train for
      that is to play against yourself: the policy learns from games against earlier copies of itself, so its opponent improves exactly as fast as it
      does. That is the last rung of this project's ladder, and the method behind the self-taught game players of the last decade.
    </p>
  </article>
</template>
