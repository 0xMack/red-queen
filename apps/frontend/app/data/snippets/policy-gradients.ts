import type { Snippet } from "~/types/code"

// Policy Gradients' snippets. The agent is Rust (libs/rl/rust/core/src/pg.rs, with the loss's gradient written out by
// hand); libs/rl/tests/reference_pg.py is the oracle -- the loss "written as the formulas read" on libs/autodiff
// tensors and differentiated by it. Both are real code.
const RUST = "libs/rl/rust/core/src/pg.rs"
const ORACLE = "libs/rl/tests/reference_pg.py"

export const reinforce: Snippet = {
  pseudo: `for each move (s, a) played, with advantage A
    π ← softmax(policy_network(s))                  // move probabilities
    loss ← loss − A × log π(a | s)                  // push the move up in proportion to what followed
    loss ← loss − β × entropy(π)                    // a small bonus for keeping options open
minimise the loss by gradient descent`,
  python: {
    source: ORACLE,
    code: `# log π(a | s) for the move that was played: a log-softmax of the network's outputs
shifted = x - Tensor(x.data.max(axis=1, keepdims=True))   # a constant shift: log-softmax is invariant to it
log_probs_all = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
log_p = log_probs_all[(rows, actions.astype(int))]
entropy = -(log_probs_all.exp() * log_probs_all).sum(axis=1)

# REINFORCE / A2C: push log π(a | s) up in proportion to the advantage of what followed,
# plus a small entropy bonus, so the policy doesn't commit before it has explored
surrogate = log_p * Tensor(advantages)
loss = surrogate.sum() * (-1.0 / n) - entropy.sum() * (entropy_coef / n)`,
  },
  rust: {
    source: RUST,
    code: `// log π(a | s) and its gradient w.r.t. the network's outputs (the logits of a softmax)
let logp = log_softmax(row);
let log_p = logp[a];                               // a: the move that was played
let d_log_p: Vec<f64> = (0..outputs).map(|j| (j == a) as u8 as f64 - p[j]).collect();

// REINFORCE / A2C: push log π(a | s) up in proportion to the advantage of what followed
loss -= advantage * log_p * scale;
// ...plus a small entropy bonus, so the policy doesn't commit before it has explored
loss -= entropy_coef * entropy * scale;`,
  },
}

export const gae: Snippet = {
  pseudo: `// GAE(λ), walking the moves backwards
running ← 0
for t from the last move down to the first
    if a new game starts after t then running ← 0
    next ← 0 if the game ended at t, otherwise V(s_{t+1})    // a game that ended has no future
    δ ← r_t + γ × next − V(s_t)                              // one step's surprise
    running ← δ + γλ × running
    A_t ← running`,
  python: {
    source: ORACLE,
    code: `def gae(rewards, values, next_values, done, cut, gamma, lam):
    """Generalized advantage estimation, as Schulman et al. (2016) define it, one episode segment at a time."""
    advantages = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if cut[t]:
            running = 0.0                                  # a new episode starts after t
        next_value = 0.0 if done[t] else next_values[t]    # a game that ended has no future
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, [a + v for a, v in zip(advantages, values)]`,
  },
  rust: {
    source: RUST,
    code: `/// GAE(λ): δ_t = r_t + γ V(s_t+1) − V(s_t),  A_t = δ_t + γλ A_t+1, restarting at every episode boundary.
for t in (0..n).rev() {
    let bootstrap = if done[t] { 0.0 } else { next_values[t] };   // a game that ended has no future
    let delta = rewards[t] + gamma * bootstrap - values[t];
    if cut[t] {
        next_advantage = 0.0;                                     // a new episode starts after t
    }
    next_advantage = delta + gamma * lambda * next_advantage;
    advantages[t] = next_advantage;
}
// λ = 1 and V = 0: plain Monte-Carlo returns (REINFORCE). λ = 1 and a learned V: returns minus a baseline.`,
  },
}

export const ppo: Snippet = {
  pseudo: `r ← π_new(a | s) / π_old(a | s)                  // how much more likely the move is than when it was played
objective ← min( r × A,  clip(r, 1 − ε, 1 + ε) × A ) // ε = 0.2: the pessimistic of the two
// once a move is 20% more (or less) likely than it was, this sample stops pushing it further`,
  python: {
    source: ORACLE,
    code: `ratio = (log_p - Tensor(old_log_probs)).exp()     # how much more likely the move is than when it was played
# min(r A, clip(r) A): the clipped term is a constant where it's the smaller one, so pick per sample
clipped_ratio = np.clip(ratio.data, 1 - clip, 1 + clip)
use_unclipped = ratio.data * advantages <= clipped_ratio * advantages
surrogate = ratio * Tensor(advantages * use_unclipped) + Tensor(clipped_ratio * advantages * ~use_unclipped)`,
  },
  rust: {
    source: RUST,
    code: `let ratio = exp(log_p - old_log_prob);             // how much more likely the move is than when it was played
let clamped = ratio.clamp(1.0 - eps, 1.0 + eps);   // eps = 0.2
let (unclipped, capped) = (ratio * advantage, clamped * advantage);
loss -= unclipped.min(capped) * scale;             // the pessimistic of the two
// the gradient flows only through the unclipped term, and only when it is the smaller one:
// once a move is 20% more (or less) likely than it was, this sample stops pushing it further
let weight = if unclipped <= capped { ratio * advantage } else { 0.0 };`,
  },
}
