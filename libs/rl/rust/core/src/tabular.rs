//! Tabular Q-learning and SARSA (docs/design/0010 Phase 1): one value per (state, action), learned from experience.
//!
//! - **State.** A `Discretizer` turns an observation into a row: the bit pattern of a binary observation (Snake's
//!   `features.v1`: 11 bits, 2,048 rows), or a grid of bins over a continuous one (Reach1D). Environments offer
//!   one when their observation is small and discrete enough (`Env::discretizer`).
//! - **Actions.** A discrete space is used as is; a continuous one is cut into `action_bins` evenly spaced values.
//! - **Learning.** One update rule covers Q-learning (off-policy: bootstrap from the best next action) and SARSA
//!   (on-policy: from the action actually taken next), with n-step returns: after `n_step` transitions the oldest
//!   pending one is updated towards `r_0 + g*r_1 + ... + g^(n-1)*r_(n-1) + g^n * bootstrap`. A game that *ended*
//!   flushes everything with no bootstrap; one *truncated* by the step cap flushes with it (the game would have
//!   gone on). Returns are accumulated back to front (`G = r + g*G`), which `tests/reference_tabular.py` repeats
//!   exactly, so the two agree bit for bit.
//! - **Exploration.** Epsilon-greedy, epsilon decaying linearly from `epsilon_start` to `epsilon_end` over
//!   `epsilon_decay_steps` actions. Greedy ties go to the first action, like `relative3.v1`'s argmax.
//!
//! SARSA needs its *next* action to update, so it chooses it in `observe` (from the exploration stream) and returns
//! it from the following `act` -- which makes every update a pure function of the transition, replayable.

use std::collections::VecDeque;

use crate::agent::{Agent, Params, Transition};
use crate::env::{Action, ActionSpace};
use crate::rng::Rng;

/// Observation -> table row.
#[derive(Clone, Debug, PartialEq)]
pub enum Discretizer {
    /// Each observation value is 0 or nonzero; value `i` is bit `i` of the row (`sum(obs[i] != 0) << i`) -- the dot
    /// product with `[1, 2, 4, ...]` a model package computes.
    Binary { bits: usize },
    /// Each dimension cut into `n` equal bins over `[low, high]` (values outside are clamped into the end bins);
    /// the row is the mixed-radix number of the bin indices, first dimension least significant.
    Bins { dims: Vec<(f64, f64, usize)> },
}

impl Discretizer {
    pub fn states(&self) -> usize {
        match self {
            Discretizer::Binary { bits } => 1 << bits,
            Discretizer::Bins { dims } => dims.iter().map(|&(_, _, n)| n).product(),
        }
    }

    pub fn index(&self, observation: &[f64]) -> usize {
        match self {
            Discretizer::Binary { bits } => {
                debug_assert_eq!(observation.len(), *bits);
                observation
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if v != 0.0 { 1 << i } else { 0 })
                    .sum()
            }
            Discretizer::Bins { dims } => {
                let (mut index, mut radix) = (0, 1);
                for (&value, &(low, high, n)) in observation.iter().zip(dims) {
                    let t = (value - low) / (high - low) * n as f64;
                    let bin = if t < 0.0 { 0 } else { (t as usize).min(n - 1) };
                    index += bin * radix;
                    radix *= n;
                }
                index
            }
        }
    }

    pub fn to_json(&self) -> String {
        match self {
            Discretizer::Binary { bits } => format!(r#"{{"kind": "binary", "bits": {bits}}}"#),
            Discretizer::Bins { dims } => {
                let dims: Vec<String> = dims
                    .iter()
                    .map(|(lo, hi, n)| format!("[{lo:?}, {hi:?}, {n}]"))
                    .collect();
                format!(r#"{{"kind": "bins", "dims": [{}]}}"#, dims.join(", "))
            }
        }
    }
}

/// One tabular update, as `QTableAgent::learn` receives it: states as table rows, the next action only for SARSA.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub state: usize,
    pub action: usize,
    pub reward: f64,
    pub next_state: usize,
    pub done: bool,
    pub truncated: bool,
    pub next_action: Option<usize>,
}

pub struct QTableAgent {
    sarsa: bool,
    discretizer: Discretizer,
    /// A continuous space's action values (`action_bins` of them); `None` for a discrete space.
    action_values: Option<Vec<f64>>,
    actions: usize,
    q: Vec<f64>,
    visits: Vec<u32>,
    alpha: f64,
    gamma: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_steps: f64,
    n_step: usize,
    acted: u64,
    pending: VecDeque<(usize, usize, f64)>,
    /// SARSA's next action, chosen in `observe` and returned by the next `act`.
    queued: Option<usize>,
    td_abs_sum: f64,
    td_updates: u64,
}

impl QTableAgent {
    pub const PARAMS: [&'static str; 8] = [
        "alpha",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "initial_q",
        "n_step",
        "action_bins",
    ];

    pub fn new(sarsa: bool, discretizer: Discretizer, space: ActionSpace, params: &Params) -> Result<Self, String> {
        let name = if sarsa { "sarsa" } else { "q_learning" };
        params.check(name, &Self::PARAMS)?;
        let (actions, action_values) = match space {
            ActionSpace::Discrete(n) => (n, None),
            ActionSpace::Continuous { low, high } => {
                let bins = params.get("action_bins", 3.0) as usize;
                if bins < 2 {
                    return Err("action_bins must be at least 2".into());
                }
                let values = (0..bins)
                    .map(|i| low + (high - low) * i as f64 / (bins - 1) as f64)
                    .collect();
                (bins, Some(values))
            }
        };
        let n_step = params.get("n_step", 1.0);
        let (alpha, gamma) = (params.get("alpha", 0.1), params.get("gamma", 0.95));
        if n_step < 1.0 || n_step.fract() != 0.0 {
            return Err(format!("n_step must be a positive integer, got {n_step}"));
        }
        if !(alpha > 0.0 && alpha <= 1.0 && (0.0..=1.0).contains(&gamma)) {
            return Err(format!(
                "need 0 < alpha <= 1 and 0 <= gamma <= 1, got alpha {alpha}, gamma {gamma}"
            ));
        }
        let states = discretizer.states();
        Ok(QTableAgent {
            sarsa,
            q: vec![params.get("initial_q", 0.0); states * actions],
            visits: vec![0; states],
            discretizer,
            action_values,
            actions,
            alpha,
            gamma,
            epsilon_start: params.get("epsilon_start", 1.0),
            epsilon_end: params.get("epsilon_end", 0.05),
            epsilon_decay_steps: params.get("epsilon_decay_steps", 100_000.0),
            n_step: n_step as usize,
            acted: 0,
            pending: VecDeque::new(),
            queued: None,
            td_abs_sum: 0.0,
            td_updates: 0,
        })
    }

    pub fn epsilon(&self) -> f64 {
        let t = self.acted as f64 / self.epsilon_decay_steps.max(1.0);
        if t >= 1.0 {
            return self.epsilon_end; // exactly: start + (end - start) * 1 isn't always `end` in floating point
        }
        self.epsilon_start + (self.epsilon_end - self.epsilon_start) * t
    }

    pub fn table(&self) -> &[f64] {
        &self.q
    }

    fn row(&self, state: usize) -> &[f64] {
        &self.q[state * self.actions..(state + 1) * self.actions]
    }

    fn greedy(&self, state: usize) -> usize {
        let row = self.row(state);
        let mut best = 0;
        for (i, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = i;
            }
        }
        best
    }

    fn epsilon_greedy(&self, state: usize, rng: &mut Rng) -> usize {
        if rng.uniform() < self.epsilon() {
            rng.below(self.actions as u32) as usize
        } else {
            self.greedy(state)
        }
    }

    fn to_action(&self, index: usize) -> Action {
        match &self.action_values {
            None => Action::Discrete(index),
            Some(values) => Action::Continuous(values[index]),
        }
    }

    fn action_index(&self, action: Action) -> usize {
        match (action, &self.action_values) {
            (Action::Discrete(i), None) => i,
            (Action::Continuous(v), Some(values)) => {
                values.iter().position(|&x| x == v).expect("an action this agent chose")
            }
            (other, _) => panic!("action {other:?} doesn't match this table's action space"),
        }
    }

    /// Move the oldest pending transition towards its n-step return ending in `bootstrap`.
    fn update_oldest(&mut self, bootstrap: f64) {
        let mut g = bootstrap;
        for &(_, _, reward) in self.pending.iter().rev() {
            g = reward + self.gamma * g;
        }
        let (state, action, _) = self.pending.pop_front().expect("something pending");
        let cell = state * self.actions + action;
        let td = g - self.q[cell];
        self.q[cell] += self.alpha * td;
        self.visits[state] += 1;
        self.td_abs_sum += td.abs();
        self.td_updates += 1;
    }

    fn flush(&mut self, bootstrap: f64) {
        while !self.pending.is_empty() {
            self.update_oldest(bootstrap);
        }
    }

    /// One transition's learning, in table terms. `next_action` is SARSA's (ignored by Q-learning).
    pub fn learn(&mut self, step: Step) {
        self.pending.push_back((step.state, step.action, step.reward));
        let value_next = |agent: &Self| match (agent.sarsa, step.next_action) {
            (true, Some(a)) => agent.row(step.next_state)[a],
            _ => {
                let row = agent.row(step.next_state);
                row[agent.greedy(step.next_state)]
            }
        };
        if step.done {
            self.flush(0.0);
        } else if step.truncated {
            let bootstrap = value_next(self);
            self.flush(bootstrap);
        } else if self.pending.len() == self.n_step {
            let bootstrap = value_next(self);
            self.update_oldest(bootstrap);
        }
    }

    /// The policy as a JSON table (`modelpack.champions.QTable`): rows in `Discretizer` order, one value per action.
    pub fn snapshot_json(&self, algorithm: &str) -> String {
        let actions = match &self.action_values {
            None => format!(r#"{{"kind": "discrete", "count": {}}}"#, self.actions),
            Some(values) => format!(r#"{{"kind": "continuous", "values": {values:?}}}"#),
        };
        let values: Vec<String> = self.q.iter().map(|v| format!("{v:?}")).collect();
        format!(
            r#"{{"type": "qtable", "algorithm": "{algorithm}", "discretizer": {}, "actions": {actions}, "values": [{}]}}"#,
            self.discretizer.to_json(),
            values.join(", ")
        )
    }
}

impl Agent for QTableAgent {
    fn name(&self) -> &'static str {
        if self.sarsa {
            "sarsa"
        } else {
            "q_learning"
        }
    }

    fn act(&mut self, observation: &[f64], rng: &mut Rng) -> Action {
        let index = match self.queued.take() {
            Some(queued) => queued,
            None => self.epsilon_greedy(self.discretizer.index(observation), rng),
        };
        self.acted += 1;
        self.to_action(index)
    }

    fn act_greedy(&mut self, observation: &[f64], _rng: &mut Rng) -> Action {
        self.to_action(self.greedy(self.discretizer.index(observation)))
    }

    fn observe(&mut self, t: &Transition, rng: &mut Rng) {
        let next_state = self.discretizer.index(t.next_observation);
        let ends = t.done || t.truncated;
        let next_action = (self.sarsa && !ends).then(|| self.epsilon_greedy(next_state, rng));
        self.queued = next_action;
        self.learn(Step {
            state: self.discretizer.index(t.observation),
            action: self.action_index(t.action),
            reward: t.reward,
            next_state,
            done: t.done,
            truncated: t.truncated,
            next_action,
        });
    }

    /// The entropy of the epsilon-greedy policy: the greedy action has `1 - eps + eps/n`, each other `eps/n`.
    fn entropy(&self) -> f64 {
        let (eps, n) = (self.epsilon(), self.actions as f64);
        let (greedy, other) = (1.0 - eps + eps / n, eps / n);
        let term = |p: f64| if p > 0.0 { -p * libm::log(p) } else { 0.0 };
        term(greedy) + (n - 1.0) * term(other)
    }

    fn extras(&mut self) -> Vec<(String, f64)> {
        let visited = self.visits.iter().filter(|&&v| v > 0).count();
        let mean_td = if self.td_updates > 0 {
            self.td_abs_sum / self.td_updates as f64
        } else {
            0.0
        };
        self.td_abs_sum = 0.0;
        self.td_updates = 0;
        vec![
            ("epsilon".into(), self.epsilon()),
            ("mean_abs_td_error".into(), mean_td),
            ("states_visited".into(), visited as f64),
            ("state_coverage".into(), visited as f64 / self.visits.len() as f64),
        ]
    }

    fn snapshot(&self) -> String {
        self.snapshot_json(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Trainer, TrainerConfig};
    use crate::env::{Env, EnvFactory, Step as EnvStep};
    use crate::rng::Stream;

    /// A corridor of `LEN` cells: start at 0, +1 for reaching the end; moving left from 0 stays put. Observation:
    /// the position as bits. The optimal policy is "always right".
    const LEN: usize = 6;
    struct Corridor {
        at: usize,
    }
    impl Env for Corridor {
        fn observation_size(&self) -> usize {
            3
        }
        fn action_space(&self) -> ActionSpace {
            ActionSpace::Discrete(2)
        }
        fn reset(&mut self, _seed: u64) -> Vec<f64> {
            self.at = 0;
            self.obs()
        }
        fn step(&mut self, action: Action) -> EnvStep {
            match action {
                Action::Discrete(1) => self.at += 1,
                _ => self.at = self.at.saturating_sub(1),
            }
            let done = self.at == LEN - 1;
            EnvStep {
                observation: self.obs(),
                reward: if done { 1.0 } else { 0.0 },
                done,
            }
        }
        fn score(&self) -> f64 {
            self.at as f64
        }
        fn discretizer(&self) -> Option<Discretizer> {
            Some(Discretizer::Binary { bits: 3 })
        }
    }
    impl Corridor {
        fn obs(&self) -> Vec<f64> {
            (0..3).map(|i| ((self.at >> i) & 1) as f64).collect()
        }
    }
    struct CorridorFactory;
    impl EnvFactory for CorridorFactory {
        fn make(&self) -> Box<dyn Env> {
            Box::new(Corridor { at: 0 })
        }
        fn id(&self) -> String {
            "corridor".into()
        }
    }

    fn train(algorithm: &str, extra: &[(&str, f64)]) -> Trainer {
        let mut params: Vec<(String, f64)> = vec![("epsilon_decay_steps".into(), 3000.0)];
        params.extend(extra.iter().map(|&(k, v)| (k.to_string(), v)));
        let config = TrainerConfig {
            seed: 3,
            seed_pool: (0, 10),
            max_episode_steps: 50,
        };
        let mut trainer = Trainer::build(Box::new(CorridorFactory), algorithm, &Params::new(params), config).unwrap();
        trainer.train(6000);
        trainer
    }

    #[test]
    fn discretizers_index_and_count() {
        let binary = Discretizer::Binary { bits: 3 };
        assert_eq!(binary.states(), 8);
        assert_eq!(binary.index(&[1.0, 0.0, 1.0]), 5);
        let bins = Discretizer::Bins {
            dims: vec![(-1.0, 1.0, 4), (0.0, 10.0, 5)],
        };
        assert_eq!(bins.states(), 20);
        assert_eq!(bins.index(&[-1.0, 0.0]), 0);
        assert_eq!(bins.index(&[-0.4, 0.0]), 1);
        assert_eq!(
            bins.index(&[5.0, 9.99]),
            3 + 4 * 4,
            "clamped high in dim 0, last bin in dim 1"
        );
        assert_eq!(bins.index(&[-9.0, -9.0]), 0, "clamped low");
        assert!(bins.to_json().contains(r#""kind": "bins""#));
    }

    #[test]
    fn q_learning_sarsa_and_n_step_all_learn_the_corridor() {
        for (algorithm, extra) in [
            ("q_learning", vec![]),
            ("sarsa", vec![]),
            ("q_learning", vec![("n_step", 3.0)]),
            ("sarsa", vec![("n_step", 4.0)]),
        ] {
            let mut trainer = train(algorithm, &extra);
            let episode = trainer.evaluate(&[0], 20)[0];
            assert_eq!(
                episode.steps as usize,
                LEN - 1,
                "{algorithm} {extra:?} walks straight to the end"
            );
            assert_eq!(episode.total_reward, 1.0);
        }
    }

    #[test]
    fn epsilon_decays_and_entropy_follows() {
        let space = ActionSpace::Discrete(3);
        let params = Params::new([("epsilon_decay_steps".to_string(), 10.0)]);
        let mut agent = QTableAgent::new(false, Discretizer::Binary { bits: 2 }, space, &params).unwrap();
        assert!((agent.entropy() - libm::log(3.0)).abs() < 1e-12, "epsilon 1: uniform");
        let mut rng = Rng::new(0, Stream::Explore);
        for _ in 0..20 {
            agent.act(&[0.0, 0.0], &mut rng);
        }
        assert!((agent.epsilon() - 0.05).abs() < 1e-12);
        assert!(agent.entropy() < 0.3);
        let extras = agent.extras();
        assert_eq!(extras[0], ("epsilon".to_string(), 0.05));
    }

    #[test]
    fn n_step_returns_are_what_the_formula_says() {
        // gamma 0.5, alpha 1: after the third transition the oldest cell holds r0 + g r1 + g^2 r2 + g^3 max Q(s3).
        let params = Params::new([
            ("alpha".to_string(), 1.0),
            ("gamma".to_string(), 0.5),
            ("n_step".to_string(), 3.0),
            ("initial_q".to_string(), 2.0),
        ]);
        let mut agent = QTableAgent::new(
            false,
            Discretizer::Binary { bits: 2 },
            ActionSpace::Discrete(2),
            &params,
        )
        .unwrap();
        let step = |state, reward, next_state| Step {
            state,
            action: 0,
            reward,
            next_state,
            done: false,
            truncated: false,
            next_action: None,
        };
        agent.learn(step(0, 1.0, 1));
        agent.learn(step(1, 2.0, 2));
        assert_eq!(agent.table()[0], 2.0, "nothing updated before n transitions");
        agent.learn(step(2, 4.0, 3));
        assert_eq!(agent.table()[0], 1.0 + 0.5 * 2.0 + 0.25 * 4.0 + 0.125 * 2.0);
        // a game over flushes the rest with no bootstrap
        agent.learn(Step {
            done: true,
            ..step(3, 8.0, 0)
        });
        assert_eq!(agent.table()[2], 2.0 + 0.5 * 4.0 + 0.25 * 8.0);
        assert_eq!(agent.table()[6], 8.0);
    }

    #[test]
    fn continuous_actions_are_binned_and_bad_params_rejected() {
        let space = ActionSpace::Continuous { low: -1.0, high: 1.0 };
        let params = Params::new([("action_bins".to_string(), 5.0)]);
        let agent = QTableAgent::new(false, Discretizer::Binary { bits: 1 }, space, &params).unwrap();
        assert_eq!(agent.to_action(4), Action::Continuous(1.0));
        assert_eq!(agent.to_action(1), Action::Continuous(-0.5));
        assert!(agent.snapshot_json("q_learning").contains(r#""kind": "continuous""#));
        let bad = |k: &str, v: f64| {
            QTableAgent::new(
                false,
                Discretizer::Binary { bits: 1 },
                space,
                &Params::new([(k.to_string(), v)]),
            )
            .is_err()
        };
        assert!(bad("n_step", 0.0) && bad("n_step", 1.5) && bad("alpha", 0.0) && bad("gamma", 1.5));
        assert!(bad("temperature", 1.0), "unknown parameter");
    }
}
