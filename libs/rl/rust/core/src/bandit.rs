//! Bandit strategies (docs/design/0011): online learners for a multi-armed bandit -- they learn *within* one game,
//! pull by pull, and start every game knowing nothing. (A `Trainer` learns across episodes; a bandit's arms are
//! drawn afresh every game, so there is nothing to carry over.) Never names a game: the envs crate couples these to
//! the games core's `Bandit`.
//!
//! Every strategy keeps a table of `rows x arms`: one row per situation it can tell apart (one for a plain bandit,
//! one per lamp colour for a contextual one) -- the same shape as a Q-table, which is the point.
//!
//! - `random` -- the floor.
//! - `greedy` -- always the arm with the best estimate so far (ties at random). `alpha` 0 averages; > 0 is a
//!   constant step size (recent payouts count more).
//! - `epsilon_greedy` -- greedy, but a random arm with probability `epsilon` (falling linearly to 0 over `decay`
//!   pulls if `decay` > 0).
//! - `optimistic` -- greedy on `(initial + sum of payouts) / (1 + pulls)`: every arm starts with one imaginary pull
//!   that paid `initial` (by default the game's largest payout), so untried arms look best until tried.
//! - `ucb1` -- untried arms first, then `estimate + c * scale * sqrt(ln t / n)`: exploration as an uncertainty bonus.
//! - `thompson` -- sample each arm's belief and pull the best sample. Win/lose payouts: a Beta(1 + wins, 1 + losses)
//!   belief, sampled exactly as the `(1 + wins)`-th smallest of `1 + pulls` uniforms (integer shapes need no
//!   transcendental). Other payouts: untried first, then a normal around the average with spread `scale / sqrt(n)`.
//! - `gradient` -- preferences, not values: a softmax over preferences `H`, and after each payout
//!   `H_b += alpha (r - baseline) (1[b = a] - pi_b)`, the baseline the average payout so far -- REINFORCE with a
//!   baseline, in one row.
//! - `q_table` -- `tabular::QTableAgent` itself, unchanged, with `gamma = 0` and a constant epsilon: the Q-learning
//!   chapter's agent, playing a bandit.

use crate::agent::{Agent, Params, Transition};
use crate::env::{Action, ActionSpace};
use crate::rng::Rng;
use crate::tabular::{Discretizer, QTableAgent};

/// What a strategy may know about the game before it starts: its shape, and the payouts' kind and scale (a player
/// is told the rules, not the arms).
#[derive(Clone, Copy, Debug)]
pub struct Hints {
    pub arms: usize,
    /// Situations the strategy can tell apart: 1, or 2 when it sees the lamp.
    pub rows: usize,
    /// Every payout is 0 or 1.
    pub binary: bool,
    /// The payouts' typical spread.
    pub scale: f64,
    /// The largest payout an arm can make.
    pub max_payout: f64,
}

/// What a strategy currently believes about each arm in one row -- what a demo draws.
#[derive(Clone, Debug, PartialEq)]
pub struct Beliefs {
    /// Its value estimate per arm (for `gradient`, the preference).
    pub values: Vec<f64>,
    /// How unsure it is, per arm (UCB's bonus, a posterior's sd); 0 where the strategy has no notion of it.
    pub spread: Vec<f64>,
    pub counts: Vec<u32>,
    /// The probability of pulling each arm next, for strategies that choose by chance (`gradient`); empty otherwise.
    pub probabilities: Vec<f64>,
}

pub trait BanditStrategy {
    fn name(&self) -> &'static str;
    /// The arm to pull in situation `row`.
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize;
    /// Learn from arm `arm` paying `reward` in situation `row`.
    fn update(&mut self, row: usize, arm: usize, reward: f64);
    fn beliefs(&self, row: usize) -> Beliefs;
}

pub const STRATEGIES: [&str; 8] = [
    "random",
    "greedy",
    "epsilon_greedy",
    "optimistic",
    "ucb1",
    "thompson",
    "gradient",
    "q_table",
];

/// Uniformly among the arms with the largest `score` (exact ties only).
fn argmax_random(scores: &[f64], rng: &mut Rng) -> usize {
    let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ties: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] == best).collect();
    if ties.len() == 1 {
        ties[0]
    } else {
        ties[rng.below(ties.len() as u32) as usize]
    }
}

/// Per-cell pull counts and running estimates -- the table most strategies keep.
#[derive(Clone, Debug)]
struct Table {
    arms: usize,
    counts: Vec<u32>,
    values: Vec<f64>,
    sums: Vec<f64>,
    /// Constant step size; 0 = the sample average.
    alpha: f64,
}

impl Table {
    fn new(hints: &Hints, alpha: f64, initial: f64) -> Table {
        let cells = hints.rows * hints.arms;
        Table {
            arms: hints.arms,
            counts: vec![0; cells],
            values: vec![initial; cells],
            sums: vec![0.0; cells],
            alpha,
        }
    }

    fn cell(&self, row: usize, arm: usize) -> usize {
        row * self.arms + arm
    }

    fn row<'a, T>(&self, data: &'a [T], row: usize) -> &'a [T] {
        &data[row * self.arms..(row + 1) * self.arms]
    }

    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        let c = self.cell(row, arm);
        self.counts[c] += 1;
        self.sums[c] += reward;
        let step = if self.alpha > 0.0 {
            self.alpha
        } else {
            1.0 / self.counts[c] as f64
        };
        self.values[c] += step * (reward - self.values[c]);
    }

    fn pulls(&self, row: usize) -> u32 {
        self.row(&self.counts, row).iter().sum()
    }

    fn untried(&self, row: usize) -> Vec<f64> {
        self.row(&self.counts, row)
            .iter()
            .map(|&n| if n == 0 { 1.0 } else { 0.0 })
            .collect()
    }

    fn beliefs(&self, row: usize, values: Vec<f64>, spread: Vec<f64>) -> Beliefs {
        Beliefs {
            values,
            spread,
            counts: self.row(&self.counts, row).to_vec(),
            probabilities: Vec::new(),
        }
    }

    fn plain(&self, row: usize) -> Beliefs {
        self.beliefs(row, self.row(&self.values, row).to_vec(), vec![0.0; self.arms])
    }
}

struct Random {
    table: Table,
}

impl BanditStrategy for Random {
    fn name(&self) -> &'static str {
        "random"
    }
    fn choose(&mut self, _row: usize, rng: &mut Rng) -> usize {
        rng.below(self.table.arms as u32) as usize
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        self.table.update(row, arm, reward);
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        self.table.plain(row)
    }
}

/// Greedy and epsilon-greedy (greedy is epsilon 0).
struct EpsilonGreedy {
    name: &'static str,
    table: Table,
    epsilon: f64,
    decay: f64,
    chosen: u64,
}

impl EpsilonGreedy {
    fn epsilon(&self) -> f64 {
        if self.decay > 0.0 {
            self.epsilon * (1.0 - self.chosen as f64 / self.decay).max(0.0)
        } else {
            self.epsilon
        }
    }
}

impl BanditStrategy for EpsilonGreedy {
    fn name(&self) -> &'static str {
        self.name
    }
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
        let epsilon = self.epsilon();
        self.chosen += 1;
        if epsilon > 0.0 && rng.uniform() < epsilon {
            return rng.below(self.table.arms as u32) as usize;
        }
        argmax_random(self.table.row(&self.table.values, row), rng)
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        self.table.update(row, arm, reward);
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        self.table.plain(row)
    }
}

struct Optimistic {
    table: Table,
    initial: f64,
}

impl Optimistic {
    fn estimates(&self, row: usize) -> Vec<f64> {
        let (counts, sums) = (
            self.table.row(&self.table.counts, row),
            self.table.row(&self.table.sums, row),
        );
        (0..self.table.arms)
            .map(|a| (self.initial + sums[a]) / (1.0 + counts[a] as f64))
            .collect()
    }
}

impl BanditStrategy for Optimistic {
    fn name(&self) -> &'static str {
        "optimistic"
    }
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
        argmax_random(&self.estimates(row), rng)
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        self.table.update(row, arm, reward);
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        self.table.beliefs(row, self.estimates(row), vec![0.0; self.table.arms])
    }
}

struct Ucb1 {
    table: Table,
    c: f64,
    scale: f64,
}

impl Ucb1 {
    fn bonuses(&self, row: usize) -> Vec<f64> {
        let t = self.table.pulls(row).max(1) as f64;
        self.table
            .row(&self.table.counts, row)
            .iter()
            .map(|&n| {
                if n == 0 {
                    f64::INFINITY
                } else {
                    self.c * self.scale * libm::sqrt(libm::log(t) / n as f64)
                }
            })
            .collect()
    }
}

impl BanditStrategy for Ucb1 {
    fn name(&self) -> &'static str {
        "ucb1"
    }
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
        let untried = self.table.untried(row);
        if untried.contains(&1.0) {
            return argmax_random(&untried, rng);
        }
        let values = self.table.row(&self.table.values, row);
        let scores: Vec<f64> = self.bonuses(row).iter().zip(values).map(|(b, v)| v + b).collect();
        argmax_random(&scores, rng)
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        self.table.update(row, arm, reward);
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        let spread = self
            .bonuses(row)
            .iter()
            .map(|&b| if b.is_finite() { b } else { 0.0 })
            .collect();
        self.table
            .beliefs(row, self.table.row(&self.table.values, row).to_vec(), spread)
    }
}

struct Thompson {
    table: Table,
    binary: bool,
    scale: f64,
}

/// A draw from Beta(a, b) for positive integers a, b: the a-th smallest of a + b - 1 uniforms.
pub fn beta_order_statistic(a: u32, b: u32, rng: &mut Rng) -> f64 {
    let mut draws: Vec<f64> = (0..a + b - 1).map(|_| rng.uniform()).collect();
    draws.sort_by(|x, y| x.partial_cmp(y).unwrap());
    draws[a as usize - 1]
}

impl Thompson {
    /// (1 + wins, 1 + losses) of a binary arm.
    fn shapes(&self, row: usize, arm: usize) -> (u32, u32) {
        let c = self.table.cell(row, arm);
        let wins = self.table.sums[c] as u32;
        (1 + wins, 1 + self.table.counts[c] - wins)
    }
}

impl BanditStrategy for Thompson {
    fn name(&self) -> &'static str {
        "thompson"
    }
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
        let samples: Vec<f64> = if self.binary {
            (0..self.table.arms)
                .map(|a| {
                    let (alpha, beta) = self.shapes(row, a);
                    beta_order_statistic(alpha, beta, rng)
                })
                .collect()
        } else {
            let untried = self.table.untried(row);
            if untried.contains(&1.0) {
                return argmax_random(&untried, rng);
            }
            let (counts, values) = (
                self.table.row(&self.table.counts, row),
                self.table.row(&self.table.values, row),
            );
            (0..self.table.arms)
                .map(|a| values[a] + self.scale / libm::sqrt(counts[a] as f64) * rng.normal())
                .collect()
        };
        argmax_random(&samples, rng)
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        self.table.update(row, arm, reward);
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        let (values, spread) = (0..self.table.arms)
            .map(|a| {
                if self.binary {
                    let (alpha, beta) = self.shapes(row, a);
                    let (alpha, beta) = (alpha as f64, beta as f64);
                    let n = alpha + beta;
                    (alpha / n, libm::sqrt(alpha * beta / (n * n * (n + 1.0))))
                } else {
                    let c = self.table.cell(row, a);
                    let n = self.table.counts[c];
                    if n == 0 {
                        (0.0, self.scale)
                    } else {
                        (self.table.values[c], self.scale / libm::sqrt(n as f64))
                    }
                }
            })
            .unzip();
        self.table.beliefs(row, values, spread)
    }
}

struct Gradient {
    table: Table,
    alpha: f64,
    use_baseline: bool,
    preferences: Vec<f64>,
    /// Per row: the average payout so far, and how many payouts it averages.
    baselines: Vec<(f64, u32)>,
}

impl Gradient {
    pub fn probabilities(&self, row: usize) -> Vec<f64> {
        let h = self.table.row(&self.preferences, row);
        let top = h.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = h.iter().map(|&x| libm::exp(x - top)).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }
}

impl BanditStrategy for Gradient {
    fn name(&self) -> &'static str {
        "gradient"
    }
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
        let (probabilities, u) = (self.probabilities(row), rng.uniform());
        let mut cumulative = 0.0;
        for (a, p) in probabilities.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return a;
            }
        }
        probabilities.len() - 1
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        self.table.update(row, arm, reward);
        let (mean, n) = self.baselines[row];
        // The baseline is the average of the payouts *before* this one (the first payout is its own baseline).
        let baseline = if !self.use_baseline {
            0.0
        } else if n == 0 {
            reward
        } else {
            mean
        };
        let probabilities = self.probabilities(row);
        let advantage = reward - baseline;
        for (b, p) in probabilities.iter().enumerate() {
            let indicator = if b == arm { 1.0 } else { 0.0 };
            self.preferences[row * self.table.arms + b] += self.alpha * advantage * (indicator - p);
        }
        self.baselines[row] = (mean + (reward - mean) / (n + 1) as f64, n + 1);
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        let mut beliefs = self.table.beliefs(
            row,
            self.table.row(&self.preferences, row).to_vec(),
            vec![0.0; self.table.arms],
        );
        beliefs.probabilities = self.probabilities(row);
        beliefs
    }
}

/// The Q-learning chapter's tabular agent, playing a bandit: a row per situation, `gamma` 0 (no next state to
/// bootstrap from), a constant epsilon, and every pull its own finished episode.
struct QTable {
    agent: QTableAgent,
    arms: usize,
    bits: usize,
    counts: Vec<u32>,
}

impl QTable {
    fn observation(&self, row: usize) -> Vec<f64> {
        (0..self.bits).map(|i| ((row >> i) & 1) as f64).collect()
    }
}

impl BanditStrategy for QTable {
    fn name(&self) -> &'static str {
        "q_table"
    }
    fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
        match self.agent.act(&self.observation(row), rng) {
            Action::Discrete(a) => a,
            other => unreachable!("a discrete table acts discretely, got {other:?}"),
        }
    }
    fn update(&mut self, row: usize, arm: usize, reward: f64) {
        let observation = self.observation(row);
        self.counts[row * self.arms + arm] += 1;
        // A different rng than `choose`'s: a done transition makes the agent draw nothing, so this one is never used.
        let mut unused = Rng::new(0, crate::rng::Stream::Data);
        self.agent.observe(
            &Transition {
                observation: &observation,
                action: Action::Discrete(arm),
                reward,
                next_observation: &observation,
                done: true,
                truncated: false,
            },
            &mut unused,
        );
    }
    fn beliefs(&self, row: usize) -> Beliefs {
        let values = self.agent.table()[row * self.arms..(row + 1) * self.arms].to_vec();
        Beliefs {
            values,
            spread: vec![0.0; self.arms],
            counts: self.counts[row * self.arms..(row + 1) * self.arms].to_vec(),
            probabilities: Vec::new(),
        }
    }
}

/// A strategy by name, for a game with `hints`. Unknown parameters are an error.
pub fn build_strategy(name: &str, hints: &Hints, params: &Params) -> Result<Box<dyn BanditStrategy>, String> {
    if hints.arms == 0 || !(hints.rows == 1 || hints.rows == 2) {
        return Err(format!("need at least one arm and 1 or 2 rows, got {hints:?}"));
    }
    let alpha = params.get("alpha", 0.0);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(format!("alpha must be in [0, 1], got {alpha}"));
    }
    Ok(match name {
        "random" => {
            params.check(name, &[])?;
            Box::new(Random {
                table: Table::new(hints, 0.0, 0.0),
            })
        }
        "greedy" | "epsilon_greedy" => {
            params.check(name, &["alpha", "initial", "epsilon", "decay"])?;
            let epsilon = if name == "greedy" {
                0.0
            } else {
                params.get("epsilon", 0.1)
            };
            if name == "greedy" && (params.get("epsilon", 0.0) != 0.0) {
                return Err("greedy never explores: use epsilon_greedy".into());
            }
            Box::new(EpsilonGreedy {
                name: if name == "greedy" { "greedy" } else { "epsilon_greedy" },
                table: Table::new(hints, alpha, params.get("initial", 0.0)),
                epsilon,
                decay: params.get("decay", 0.0),
                chosen: 0,
            })
        }
        "optimistic" => {
            params.check(name, &["initial"])?;
            Box::new(Optimistic {
                table: Table::new(hints, 0.0, 0.0),
                initial: params.get("initial", hints.max_payout),
            })
        }
        "ucb1" => {
            params.check(name, &["c", "alpha"])?;
            Box::new(Ucb1 {
                table: Table::new(hints, alpha, 0.0),
                c: params.get("c", core::f64::consts::SQRT_2),
                scale: hints.scale,
            })
        }
        "thompson" => {
            params.check(name, &[])?;
            Box::new(Thompson {
                table: Table::new(hints, 0.0, 0.0),
                binary: hints.binary,
                scale: hints.scale,
            })
        }
        "gradient" => {
            params.check(name, &["alpha", "baseline"])?;
            Box::new(Gradient {
                table: Table::new(hints, 0.0, 0.0),
                alpha: params.get("alpha", 0.1),
                use_baseline: params.get("baseline", 1.0) != 0.0,
                preferences: vec![0.0; hints.rows * hints.arms],
                baselines: vec![(0.0, 0); hints.rows],
            })
        }
        "q_table" => {
            params.check(name, &["alpha", "epsilon", "initial_q"])?;
            let epsilon = params.get("epsilon", 0.1);
            let bits = if hints.rows == 2 { 1 } else { 0 };
            let table_params = Params::new([
                ("alpha".to_string(), params.get("alpha", 0.1).max(1e-9)),
                ("gamma".to_string(), 0.0),
                ("epsilon_start".to_string(), epsilon),
                ("epsilon_end".to_string(), epsilon),
                ("initial_q".to_string(), params.get("initial_q", 0.0)),
            ]);
            let agent = QTableAgent::new(
                false,
                Discretizer::Binary { bits },
                ActionSpace::Discrete(hints.arms),
                &table_params,
            )?;
            Box::new(QTable {
                agent,
                arms: hints.arms,
                bits,
                counts: vec![0; hints.rows * hints.arms],
            })
        }
        other => return Err(format!("unknown bandit strategy {other:?} (known: {STRATEGIES:?})")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Stream;

    fn hints(arms: usize, binary: bool) -> Hints {
        Hints {
            arms,
            rows: 1,
            binary,
            scale: 1.0,
            max_payout: 1.0,
        }
    }

    /// Plays `pulls` pulls of a fixed Bernoulli bandit (means `means`) and returns the pulls per arm.
    fn play(name: &str, params: &Params, means: &[f64], pulls: usize, seed: u64) -> Vec<u32> {
        let mut strategy = build_strategy(name, &hints(means.len(), true), params).unwrap();
        let (mut rng, mut payouts) = (Rng::new(seed, Stream::Explore), Rng::new(seed, Stream::Data));
        for _ in 0..pulls {
            let arm = strategy.choose(0, &mut rng);
            let reward = if payouts.uniform() < means[arm] { 1.0 } else { 0.0 };
            strategy.update(0, arm, reward);
        }
        strategy.beliefs(0).counts
    }

    #[test]
    fn every_strategy_finds_an_obvious_best_arm_except_random_and_greedy() {
        let means = [0.1, 0.2, 0.9, 0.3];
        let share = |name| {
            let pulls: u32 = (0..20).map(|s| play(name, &Params::default(), &means, 300, s)[2]).sum();
            pulls as f64 / (20.0 * 300.0)
        };
        for name in STRATEGIES {
            let share = share(name);
            match name {
                "random" => assert!((0.2..0.3).contains(&share), "{name}: {share}"),
                // greedy commits to the first arm that pays and never looks again: the whole point of the game
                "greedy" => assert!(share < 0.75, "{name}: {share}"),
                _ => assert!(share > 0.6, "{name} pulled the best arm only {share} of the time"),
            }
        }
        assert!(share("greedy") < share("thompson") - 0.2);
    }

    #[test]
    fn beta_draws_have_the_right_mean_and_q_table_is_the_tabular_update() {
        let mut rng = Rng::new(3, Stream::Data);
        let mean: f64 = (0..4000).map(|_| beta_order_statistic(3, 7, &mut rng)).sum::<f64>() / 4000.0;
        assert!((mean - 0.3).abs() < 0.01, "{mean}");

        let params = Params::new([("alpha".to_string(), 0.5)]);
        let mut q = build_strategy("q_table", &hints(3, true), &params).unwrap();
        q.update(0, 1, 1.0);
        q.update(0, 1, 0.0);
        q.update(0, 2, 1.0);
        assert_eq!(q.beliefs(0).values, vec![0.0, 0.25, 0.5]); // Q += 0.5 (r - Q), no bootstrap
        assert_eq!(q.beliefs(0).counts, vec![0, 2, 1]);
    }

    #[test]
    fn a_contextual_strategy_keeps_one_row_per_situation() {
        let hints = Hints {
            rows: 2,
            ..hints(3, true)
        };
        for name in ["greedy", "q_table", "thompson"] {
            let mut s = build_strategy(name, &hints, &Params::default()).unwrap();
            s.update(1, 2, 1.0);
            assert_eq!(s.beliefs(0).counts, vec![0, 0, 0], "{name}");
            assert_eq!(s.beliefs(1).counts, vec![0, 0, 1], "{name}");
        }
    }

    #[test]
    fn gradient_probabilities_follow_the_preferences_and_params_are_checked() {
        let mut g = build_strategy("gradient", &hints(2, true), &Params::default()).unwrap();
        assert_eq!(g.beliefs(0).probabilities, vec![0.5, 0.5]);
        g.update(0, 0, 0.0); // first payout: its own baseline, no change
        g.update(0, 0, 1.0); // better than the baseline: arm 0 gains
        let p = g.beliefs(0).probabilities;
        assert!(p[0] > 0.5 && (p[0] + p[1] - 1.0).abs() < 1e-12);
        assert!(build_strategy("ucb1", &hints(2, true), &Params::new([("epsilon".to_string(), 0.1)])).is_err());
        assert!(build_strategy("greedy", &hints(2, true), &Params::new([("epsilon".to_string(), 0.1)])).is_err());
        assert!(build_strategy("oracle", &hints(2, true), &Params::default()).is_err());
    }
}
