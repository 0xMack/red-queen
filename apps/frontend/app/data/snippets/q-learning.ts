import type { Snippet } from "~/types/code"

// Q-learning's snippets. The agent is Rust (libs/rl/rust/core/src/tabular.rs -- natively for the experiments, as
// WASM in this page's lab); libs/rl/tests/reference_tabular.py is the plain-Python oracle whose tables must come out
// *equal* to the Rust's. Python for what the oracle doesn't need (indexing, exploring) is a translation.
const RUST = "libs/rl/rust/core/src/tabular.rs"
const ORACLE = "libs/rl/tests/reference_tabular.py"

export const discretizer: Snippet = {
  pseudo: `// Snake's features.v1: 11 numbers, each 0 or 1 -> one of 2^11 = 2,048 rows
function row_index(observation)
    index ← 0
    for each feature i that is on
        index ← index + 2^i          // read the features as the bits of one binary number
    return index`,
  python: `# Snake's features.v1: 11 numbers, each 0 or 1 -> one of 2^11 = 2,048 rows
def row_index(observation: list[float]) -> int:
    return sum(1 << i for i, v in enumerate(observation) if v != 0.0)`,
  rust: {
    source: RUST,
    code: `// Snake's features.v1: 11 numbers, each 0 or 1 -> one of 2^11 = 2,048 rows
Discretizer::Binary { bits } => observation
    .iter()
    .enumerate()
    .map(|(i, &v)| if v != 0.0 { 1 << i } else { 0 })
    .sum(),`,
  },
}

export const update: Snippet = {
  pseudo: `// Move the oldest pending move toward its n-step return
function update_oldest(bootstrap)
    G ← bootstrap                                  // the value of where the last move led
    for each pending reward r, newest first
        G ← r + γ × G                              // r + γ·(what comes after)
    (s, a) ← the oldest pending move; remove it
    Q[s, a] ← Q[s, a] + α × (G − Q[s, a])          // move a fraction α of the way`,
  python: {
    source: ORACLE,
    code: `def update_oldest(bootstrap: float) -> None:
    g = bootstrap
    for _, _, reward in reversed(pending):
        g = reward + gamma * g                     # r + γ·(what comes after)
    state, action, _ = pending.popleft()
    q[state][action] += alpha * (g - q[state][action])   # move a fraction α of the way`,
  },
  rust: {
    source: RUST,
    code: `/// Move the oldest pending transition towards its n-step return ending in \`bootstrap\`.
fn update_oldest(&mut self, bootstrap: f64) {
    let mut g = bootstrap;
    for &(_, _, reward) in self.pending.iter().rev() {
        g = reward + self.gamma * g;              // r + γ·(what comes after)
    }
    let (state, action, _) = self.pending.pop_front().expect("something pending");
    let cell = state * self.actions + action;
    let td = g - self.q[cell];                   // how wrong the table was
    self.q[cell] += self.alpha * td;             // ...move a fraction α of the way
}`,
  },
}

export const learn: Snippet = {
  pseudo: `function learn(s, a, r, s′, a′)
    append (s, a, r) to pending
    value_next ← Q[s′, a′] for SARSA (the move it will make),
                 max over b of Q[s′, b] for Q-learning (the best move)
    if the game ended then
        update every pending move with bootstrap 0             // nothing comes after
    else if the step cap cut the game off then
        update every pending move with bootstrap value_next    // the game would have gone on
    else if n moves are pending then
        update_oldest(value_next)`,
  python: {
    source: ORACLE,
    code: `for state, action, reward, next_state, done, truncated, next_action in transitions:
    pending.append((state, action, reward))
    if sarsa and next_action is not None:
        value_next = q[next_state][next_action]   # SARSA: the move it will make
    else:
        value_next = greedy_value(next_state)     # Q-learning: the best move
    if done:
        while pending:
            update_oldest(0.0)                    # the game ended: nothing comes after
    elif truncated:
        while pending:
            update_oldest(value_next)             # cut off by the step cap: the game would have gone on
    elif len(pending) == n_step:
        update_oldest(value_next)`,
  },
  rust: {
    source: RUST,
    code: `pub fn learn(&mut self, step: Step) {
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
}`,
  },
}

export const epsilon: Snippet = {
  pseudo: `function choose(s)
    with probability ε
        return any move, at random                     // explore
    return the move with the highest Q[s, ·]           // exploit

// ε falls in a straight line from 1.0 to 0.05 over the first 100k moves, then stays there.`,
  python: `def epsilon_greedy(self, state: int, rng) -> int:
    if rng.random() < self.epsilon():
        return rng.randrange(self.actions)      # explore: any move, at random
    return self.greedy(state)                   # exploit: the best move the table knows

# ε falls in a straight line from epsilon_start (1.0) to epsilon_end (0.05)
# over the first epsilon_decay_steps (100k) moves, then stays there.`,
  rust: {
    source: RUST,
    code: `fn epsilon_greedy(&self, state: usize, rng: &mut Rng) -> usize {
    if rng.uniform() < self.epsilon() {
        rng.below(self.actions as u32) as usize   // explore: any move, at random
    } else {
        self.greedy(state)                       // exploit: the best move the table knows
    }
}

// ε falls in a straight line from epsilon_start (1.0) to epsilon_end (0.05)
// over the first epsilon_decay_steps (100k) moves, then stays there.`,
  },
}
