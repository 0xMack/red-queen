<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.

const moveCode = `/// The move \`net\` would play: the one leaving the position worst for the opponent.
fn best_move(net: &Mlp, game: &Checkers) -> usize {
    let moves = game.legal_moves();
    let mut best = (0, f64::NEG_INFINITY);
    for (i, mv) in moves.iter().enumerate() {
        // simulate() encodes the position the move leaves from the *opponent's* side
        let value = -net.forward(&game.simulate(mv).expect("a legal move"))[0];
        if value > best.1 {
            best = (i, value);
        }
    }
    best.0
}`

const lambdaCode = `/// λ-return targets for one game's positions, given the network's values of them and the outcome for the
/// player who made the last move.
pub fn lambda_returns(values: &[f64], outcome: f64, lambda: f64) -> Vec<f64> {
    let n = values.len();
    let mut targets = vec![0.0; n];
    targets[n - 1] = outcome;                    // the last position: how the game actually ended, for its mover
    for t in (0..n - 1).rev() {
        // the next position belongs to the other player, so its value -- and its target -- flip sign
        targets[t] = -((1.0 - lambda) * values[t + 1] + lambda * targets[t + 1]);
    }
    targets
}`
</script>

<template>
  <article class="prose-chapter">
    <p>
      Every learner so far had something outside itself to learn from: Snake's rules and rewards, or, for the evolved Checkers players in the
      <NuxtLink to="/learn/multi-agent-games">multi-agent chapter</NuxtLink>, a fixed set of opponents to beat. This chapter removes the last of
      that. A Checkers player learns from games against <strong>itself</strong> -- no teacher, no opponents, no fitness function -- and is then
      measured against the same field as everything else.
    </p>

    <h2>Learning with nobody to learn from</h2>
    <p>
      Self-play solves the opponent problem by construction: the opponent is always exactly as strong as the learner, so there is always something to
      learn and never an opponent too strong to learn anything from. The most famous early success was <strong>TD-Gammon</strong> (Tesauro, 1992): a
      small neural network that learned backgammon purely from games against itself and reached the level of the best human players. This chapter
      builds the same algorithm for Checkers.
    </p>

    <h2>A value for every position</h2>
    <p>
      The network is a <strong>position evaluator</strong>: the board's 32 playable squares in (from the side to move: +1 for your man, +2 for your
      king, −1 and −2 for the opponent's), one number out -- how good this position is for the player about to move. To choose a move, look at the
      position each legal move leaves and pick the one that is worst for the opponent:
    </p>
    <CodeBlock lang="rust" :code="moveCode" />
    <p>
      That is exactly the kind of network the evolved Checkers players are (a 32 → 16 → 1 tanh network), used exactly the same way -- so a network
      trained by self-play drops straight into the same search (it can look several plies ahead), the same leaderboard and the same board you can
      play on.
    </p>

    <h2>TD(λ): learning from the game's own sequence</h2>
    <p>
      After each game, every position it passed through gets a target to move toward. The last position before the end gets the actual result
      (+1 if its mover went on to win, −1 if it lost, 0 for a draw). Every earlier position gets a blend of two things: the network's own estimate of
      the next position (one-step TD, λ = 0) and the target already computed for it (which ultimately leads back to the result, λ = 1). Because the
      next position belongs to the other player, both flip sign on the way back:
    </p>
    <CodeBlock lang="rust" :code="lambdaCode" />
    <p>
      That is TD(λ) -- the <NuxtLink to="/learn/q-learning">Q-learning chapter</NuxtLink>'s bootstrapping, applied to positions rather than moves,
      with λ controlling how far back each result reaches. Then one gradient step pulls every value of the game toward its target. The whole learning
      loop is these two functions and the network's backpropagation; there is no reward except who won.
    </p>

    <h2>What the dice did for backgammon</h2>
    <p>
      TD-Gammon had one enormous advantage: dice. Every game was different, so self-play automatically explored the whole game. Checkers is
      <em>deterministic</em>: two copies of the same network playing greedily play the same game every time, and learn only about that one game. So
      this learner is given its randomness on purpose: the first four moves of every game are random, and after that one move in ten (falling to
      one in fifty) is too. Without some such source of variety, self-play in a deterministic game collapses.
    </p>

    <h2>Train one, then play it</h2>
    <p>
      The same Rust training loop, compiled to WebAssembly: thirty thousand games against itself in under a minute on a desktop. Every 2,500
      games it plays 20 games against material search at the same depth (3 plies) and 20 against a random mover. Then play the result on the board
      below -- set a seat to <em>You</em>.
    </p>
    <ClientOnly>
      <SelfPlayLab />
      <template #fallback>
        <div class="card not-prose my-6 p-6 text-center text-sm text-fg-subtle">Loading the self-play lab…</div>
      </template>
    </ClientOnly>

    <h2>Against the field</h2>
    <p>
      The project trained five networks for each of four variants, 200,000 self-play games each (about three minutes), and played every final network,
      searching 3 plies, against a fixed field: material counting searched 2, 3 and 4 plies deep, and the strongest evolved evaluator on the
      leaderboard (the same 32 → 16 → 1 network, found by evolution against fixed opponents, also searching 3 plies).
    </p>
    <SelfPlayResults />
    <ul>
      <li>
        <strong>Self-play beats both the evolved evaluator and material search at the same depth</strong> (0.64 and 0.60 points per game). Only
        material search a ply deeper still wins -- the network's knowledge doesn't yet make up for seeing a move less far ahead.
      </li>
      <li>
        <strong>λ matters, and the extreme fails.</strong> Pure Monte-Carlo returns (λ = 1: every position's target is just who eventually won)
        scored 0.35, worse on every seed: a 70-move game's single result says too little about any one position in it. Bootstrapping from the
        network's own estimates of the next position (λ = 0 or 0.7) is what makes the signal usable.
      </li>
      <li>
        <strong>An opponent pool buys reliability, not strength.</strong> Playing half the games against frozen past selves left the average where it
        was (0.605 against 0.601) but shrank the spread between seeds fourfold -- the worst pure self-play run (0.45) had no counterpart. It is the
        same job the hall of fame did for the evolved players: a fixed memory of old strategies stops the population, or the network, forgetting how
        to beat them.
      </li>
    </ul>
    <p>
      One network trained with the pool, 200,000 games against itself, is on the <NuxtLink to="/games/checkers">Checkers leaderboard</NuxtLink>:
      <strong>second, at 0.81 points per game</strong> in a round robin against every entrant, behind only material search a ply deeper (0.85) and
      ahead of every evolved evaluator. You can play it there too.
    </p>

    <h2>Where this goes next</h2>
    <p>
      This closes the ladder this part of the project set out to climb: a table, a network of values, a policy learned directly, and a player that
      learns from itself. The Checkers evaluator here only ever learned values of positions and borrowed its lookahead from a fixed search. The
      natural next steps are to let the search improve the network's targets and the network guide the search -- the combination behind AlphaZero --
      and to bring the two paradigms together, evolving the hyperparameters or the architectures that the gradient methods train.
    </p>
  </article>
</template>
