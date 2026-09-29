import type { Snippet } from "~/types/code"

// Deep Q-Networks' snippets. The agent is Rust (libs/rl/rust/core/src/dqn.rs); libs/rl/tests/reference_dqn.py is
// the oracle -- the same update written with libs/autodiff tensors, whose gradients the Rust must match. The replay
// and target-network plumbing has no oracle, so its Python is a translation.
const RUST = "libs/rl/rust/core/src/dqn.rs"

export const loss: Snippet = {
  pseudo: `for each transition (s, a, G, discount, s′) in the minibatch
    // which next move: the online net picks it for Double DQN, the target net for plain DQN
    a′ ← argmax over b of Q_online(s′, b)   if double,   else argmax over b of Q_target(s′, b)
    y ← G + discount × Q_target(s′, a′)     // the target: read from the frozen TARGET network
    δ ← y − Q_online(s, a)                  // the same TD error the table used
    loss ← loss + Huber(δ)                  // quadratic near the target, linear far away
backpropagate the loss through Q_online and take an Adam step`,
  python: {
    source: "libs/rl/tests/reference_dqn.py",
    code: `next_target = q(target, next_observations).data
chooser = q(online, next_observations).data if double else next_target   # Double DQN: the online net picks
y = returns + discounts * next_target[rows, np.argmax(chooser, axis=1)]   # ...the target net values it

predicted = q_values(parts, inputs, hidden, actions, dueling, Tensor(observations))[(rows, taken)]
delta = Tensor(y) - predicted                    # the same TD error the table used
inside = np.abs(delta.data) <= 1.0
# Huber loss: quadratic near the target, linear far away -- a big surprise can't make a huge step
huber = delta * delta * (0.5 * inside) + (delta * np.sign(delta.data) - 0.5) * (~inside)
loss = (huber * weights).sum() * (1.0 / size)
loss.backward()                                  # libs/autodiff computes every gradient`,
  },
  rust: {
    source: RUST,
    code: `for i in 0..size {
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
grads = online.backward(&cache, &q_grad);`,
  },
}

export const stabilizers: Snippet = {
  pseudo: `// experience replay
store every transition in a ring buffer of the last 50,000
every 4th step: train on 32 transitions drawn at random from it

// a target network
every 2,000 steps: target ← a copy of the online network`,
  python: `# experience replay: every transition goes into a ring buffer of the last 50,000...
replay.push(stored)
# ...and every 4th step, the network trains on 32 of them drawn at random
indices, weights = replay.sample(self.batch_size, beta, self.sample_rng)

# a target network: the targets come from a frozen copy, refreshed every 2,000 steps
if self.target_update > 0 and self.observed % self.target_update == 0:
    self.target = copy.deepcopy(self.online)`,
  rust: {
    source: RUST,
    code: `// experience replay: every transition goes into a ring buffer of the last 50,000...
replay.push(&stored);
// ...and every 4th step, the network trains on 32 of them drawn at random
let (indices, weights) = replay.sample(self.batch_size, beta, &mut self.sample_rng);

// a target network: the targets come from a frozen copy, refreshed every 2,000 steps
if self.target_update > 0 && self.observed.is_multiple_of(self.target_update) {
    self.target = Some(self.online.clone());
}`,
  },
}
