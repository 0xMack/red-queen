import type { Snippet } from "~/types/code"

// Multi-Armed Bandits' snippets. The strategies run in Rust (libs/rl/rust/core/src/bandit.rs, natively and as WASM
// on this page); libs/rl/tests/reference_bandit_agents.py is the plain-Python oracle they're checked against belief
// for belief. Python for a piece the oracle doesn't spell out (choosing, sampling) is a translation.
const RUST = "libs/rl/rust/core/src/bandit.rs"
const ORACLE = "libs/rl/tests/reference_bandit_agents.py"

export const update: Snippet = {
  pseudo: `function update(row, arm, reward)
    n[arm] ← n[arm] + 1
    step ← α if a constant step is set, otherwise 1 / n[arm]   // 1/n: exactly the running average
    Q[arm] ← Q[arm] + step × (reward − Q[arm])                  // move toward what just happened`,
  python: {
    source: ORACLE,
    code: `def update(self, row, arm, reward):
    self.counts[row][arm] += 1
    # a constant step: recent payouts count more; 1/n: exactly the running average
    step = self.alpha if self.alpha else 1.0 / self.counts[row][arm]
    self.values[row][arm] += step * (reward - self.values[row][arm])   # move toward what just happened`,
  },
  rust: {
    source: RUST,
    code: `fn update(&mut self, row: usize, arm: usize, reward: f64) {
    let c = self.cell(row, arm);
    self.counts[c] += 1;
    let step = if self.alpha > 0.0 {
        self.alpha                         // a constant step: recent payouts count more
    } else {
        1.0 / self.counts[c] as f64        // 1/n: exactly the running average
    };
    self.values[c] += step * (reward - self.values[c]);   // move toward what just happened
}`,
  },
}

export const epsilon: Snippet = {
  pseudo: `function choose(row)
    with probability ε
        return any machine, at random                  // explore
    return the machine with the highest estimate       // exploit (ties broken at random)`,
  python: `def choose(self, row, rng):
    if rng.random() < self.epsilon:
        return rng.randrange(self.arms)                     # explore: any machine
    return argmax_random(self.values[row], rng)             # exploit: the best one so far`,
  rust: {
    source: RUST,
    code: `fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
    if rng.uniform() < self.epsilon() {
        return rng.below(self.arms as u32) as usize;         // explore: any machine
    }
    argmax_random(self.values(row), rng)                   // exploit: the best one so far
}`,
  },
}

export const ucb: Snippet = {
  pseudo: `function choose(row)
    if some machine is untried then return an untried machine
    t ← total pulls so far
    return the machine maximising  Q[a] + c × scale × √(ln t / n[a])   // estimate + uncertainty bonus`,
  python: {
    source: ORACLE,
    code: `# untried machines first; then the estimate plus a bonus for how little it has been tried
if 0 in counts:
    return [1.0 if n == 0 else 0.0 for n in counts]
t = sum(counts)
return [values[a] + c * self.scale * math.sqrt(math.log(t) / counts[a]) for a in range(self.arms)]`,
  },
  rust: {
    source: RUST,
    code: `fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
    let untried = self.table.untried(row);
    if untried.contains(&1.0) {
        return argmax_random(&untried, rng);             // untried machines first
    }
    // then the estimate plus a bonus for how little it has been tried: c * scale * sqrt(ln t / n)
    let values = self.table.row(&self.table.values, row);
    let scores: Vec<f64> = self.bonuses(row).iter().zip(values).map(|(b, v)| v + b).collect();
    argmax_random(&scores, rng)
}`,
  },
}

export const thompson: Snippet = {
  pseudo: `// Keep a belief per machine: Beta(1 + wins, 1 + losses)
function choose(row)
    for each machine a
        draw[a] ← a random sample from Beta(1 + wins[a], 1 + losses[a])
    return the machine with the highest draw        // plausible-and-promising gets pulled`,
  python: `def beta_order_statistic(a: int, b: int, rng) -> float:
    """A draw from Beta(a, b) for whole numbers a, b: the a-th smallest of a + b - 1 uniforms."""
    draws = sorted(rng.random() for _ in range(a + b - 1))
    return draws[a - 1]

# each pull: draw from Beta(1 + wins, 1 + losses) for every machine, pull the highest draw`,
  rust: {
    source: RUST,
    code: `/// A draw from Beta(a, b) for whole numbers a, b: the a-th smallest of a + b - 1 uniforms.
pub fn beta_order_statistic(a: u32, b: u32, rng: &mut Rng) -> f64 {
    let mut draws: Vec<f64> = (0..a + b - 1).map(|_| rng.uniform()).collect();
    draws.sort_by(|x, y| x.partial_cmp(y).unwrap());
    draws[a as usize - 1]
}
// each pull: draw from Beta(1 + wins, 1 + losses) for every machine, pull the highest draw`,
  },
}
