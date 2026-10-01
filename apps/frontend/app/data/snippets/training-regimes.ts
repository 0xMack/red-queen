import type { Snippet } from "~/types/code"

// Who to Play's snippets. Opponent sampling is Rust (libs/rl/rust/envs/src/selfplay.rs); population-based training
// is a Python job driving several Rust learners (jobs/checkers_pbt_run.py).
const RUST = "libs/rl/rust/envs/src/selfplay.rs"
const PBT = "jobs/checkers_pbt_run.py"

export const pickOpponent: Snippet = {
  pseudo: `// Before each game: itself, or a frozen past self -- which one?
function pick_opponent()
    if the pool is empty or random() ≥ pool_fraction: return itself
    if p > 0:                                  // prioritized fictitious self-play
        w[i] ← (1 − score[i])^p + 0.01 for every past self i
        return past self i with probability w[i] / Σ w
    return a past self chosen uniformly

// After a game against past self i: move its running score 5% toward the result
score[i] ← score[i] + 0.05 × (result − score[i])`,
  rust: {
    source: RUST,
    code: `fn pick_opponent(&mut self) -> PoolOpponent {
    if self.pool.is_empty() || self.explore.uniform() >= self.pool_fraction {
        return None;
    }
    let pick = if self.pfsp > 0.0 {
        let weights: Vec<f64> = self.pool_scores.iter().map(|s| (1.0 - s).powf(self.pfsp) + 0.01).collect();
        let mut target = self.explore.uniform() * weights.iter().sum::<f64>();
        let mut index = weights.len() - 1;
        for (i, w) in weights.iter().enumerate() {
            if target < *w {
                index = i;
                break;
            }
            target -= w;
        }
        index
    } else {
        self.explore.below(self.pool.len() as u32) as usize
    };
    Some((pick, self.explore.below(2) as u8))
}`,
  },
}

export const pbt: Snippet = {
  pseudo: `// Population-based training: gradients train the weights, evolution tunes the settings
members ← 8 learners, each with sampled settings (learning rate, λ, pool fraction, PFSP)
repeat
    every member trains 10,000 self-play games
    scores ← a round robin among the members (game pairs on ballot openings)
    for each member in the bottom quarter:                 // exploit
        winner ← a random member of the top quarter
        copy winner's weights and settings
        multiply each setting by 0.8 or 1.25               // explore
champion ← the round robin's best member`,
  python: {
    source: PBT,
    code: `scores = round_robin(snapshots, openings, depth, SEED_BASE + generation * 100_000)
ranked = sorted(range(members), key=lambda i: -scores[i])
if exploit and generation < generations - 1:
    quarter = max(1, members // 4)
    for loser in ranked[-quarter:]:
        winner = rng.choice(ranked[:quarter])
        trainers[loser].set_weights(list(WeightVector.from_json(snapshots[winner]).weights))
        settings[loser] = perturb(settings[winner], rng)
        for key, value in settings[loser].items():
            trainers[loser].set_param(key, value)`,
  },
}
