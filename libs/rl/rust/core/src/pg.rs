//! Policy gradients (docs/design/0010 Phase 3): learn the policy itself -- the probability of each move -- by
//! gradient ascent on the return, instead of learning values and acting greedily on them.
//!
//! One agent, three algorithms, because they share almost everything:
//!
//! - **`reinforce`** (Williams, 1992): play whole episodes, then push up the log-probability of each move in
//!   proportion to the return that followed it. Updates wait for an episode to end. `baseline=1` adds a learned
//!   state value subtracted from the return (the variance-reduction rung).
//! - **`a2c`** (advantage actor-critic): a critic V(s) is learned alongside, and advantages come from it by GAE(λ)
//!   -- bootstrapped, so an update doesn't wait for the episode to end (every `rollout_steps`). One environment here,
//!   not A3C's parallel workers: the rollout is sequential.
//! - **`ppo`** (Schulman et al., 2017): A2C's advantages, but each rollout is reused for several epochs of
//!   minibatches, with the probability ratio to the policy that collected it clipped to `1 ± clip` so no update
//!   moves the policy far. Advantages normalized per rollout.
//!
//! All three compute advantages with the same GAE: `δ_t = r_t + γ V(s_t+1) - V(s_t)`, `A_t = δ_t + γλ A_t+1`,
//! restarted at every episode boundary (a game that ended bootstraps from 0; one cut off by the step cap from the
//! critic). REINFORCE is the special case λ = 1 with V = 0 (or the learned baseline), which makes A_t the plain
//! Monte-Carlo return (minus the baseline). An entropy bonus keeps the policy from collapsing early, and gradients are
//! clipped to a global norm.
//!
//! The policy is a tanh MLP: for Snake, logits of a softmax over the three turns; for a continuous action (Reach1D),
//! the mean of a Gaussian whose log standard deviation is one more learned parameter. `pg_gradients` is the policy
//! update as a pure function -- `libs/rl/tests/reference_pg.py` checks it against `libs/autodiff`.

use crate::agent::{Agent, Params, Transition};
use crate::env::{Action, ActionSpace};
use crate::nn::{Activation, Adam, Mlp, Shape};
use crate::rng::{Rng, Stream};

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7; // 0.5 * ln(2π)

/// A rollout (or minibatch) for the policy update: `actions` are indices (discrete) or raw Gaussian samples.
#[derive(Clone, Debug, Default)]
pub struct PgBatch {
    pub observations: Vec<f64>,
    pub actions: Vec<f64>,
    pub old_log_probs: Vec<f64>,
    pub advantages: Vec<f64>,
}

pub struct PgResult {
    pub loss: f64,
    /// The policy network's gradients, then (Gaussian) the log standard deviation's.
    pub grads: Vec<f64>,
    /// Mean policy entropy (nats), mean `old - new` log-probability (an estimate of the KL divergence), and the
    /// fraction of samples whose ratio was clipped.
    pub entropy: f64,
    pub kl: f64,
    pub clip_fraction: f64,
}

fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let total = logits.iter().map(|&z| libm::exp(z - max)).sum::<f64>();
    let log_total = libm::log(total) + max;
    logits.iter().map(|&z| z - log_total).collect()
}

/// The policy loss `-mean(surrogate) - entropy_coef * mean(entropy)` and its gradient, for a softmax policy
/// (`log_std: None`) or a Gaussian one. The surrogate is `A * log π(a)` (REINFORCE, A2C) or, with `clip`, PPO's
/// `min(r A, clip(r, 1-ε, 1+ε) A)` with `r = π(a) / π_old(a)`.
pub fn pg_gradients(
    policy: &Mlp,
    log_std: Option<f64>,
    batch: &PgBatch,
    clip: Option<f64>,
    entropy_coef: f64,
) -> PgResult {
    let n = batch.actions.len();
    let outputs = policy.shape.outputs();
    let cache = policy.forward_batch(&batch.observations, n);
    let out = cache.outputs();
    let mut out_grad = vec![0.0; n * outputs];
    let mut log_std_grad = 0.0;
    let (mut loss, mut entropy_sum, mut kl_sum, mut clipped) = (0.0, 0.0, 0.0, 0usize);
    let scale = 1.0 / n as f64;
    for i in 0..n {
        let row = &out[i * outputs..(i + 1) * outputs];
        // log π(a), d log π(a) / d outputs (and / d log_std), the entropy and its gradients
        let (log_p, d_log_p, d_log_p_std, entropy, d_entropy, d_entropy_std) = match log_std {
            None => {
                let logp = log_softmax(row);
                let p: Vec<f64> = logp.iter().map(|&l| libm::exp(l)).collect();
                let a = batch.actions[i] as usize;
                let h = -p.iter().zip(&logp).map(|(&p, &l)| p * l).sum::<f64>();
                let d_log_p: Vec<f64> = (0..outputs).map(|j| (j == a) as u8 as f64 - p[j]).collect();
                let d_entropy: Vec<f64> = (0..outputs).map(|j| -p[j] * (logp[j] + h)).collect();
                (logp[a], d_log_p, 0.0, h, d_entropy, 0.0)
            }
            Some(ls) => {
                let (mean, sigma) = (row[0], libm::exp(ls));
                let z = (batch.actions[i] - mean) / sigma;
                let log_p = -0.5 * z * z - ls - HALF_LN_2PI;
                let h = ls + HALF_LN_2PI + 0.5;
                (log_p, vec![z / sigma], z * z - 1.0, h, vec![0.0], 1.0)
            }
        };
        let advantage = batch.advantages[i];
        let ratio = libm::exp(log_p - batch.old_log_probs[i]);
        // d surrogate / d log π(a)
        let weight = match clip {
            None => {
                loss -= advantage * log_p * scale;
                advantage
            }
            Some(eps) => {
                let clamped = ratio.clamp(1.0 - eps, 1.0 + eps);
                if (ratio - 1.0).abs() > eps {
                    clipped += 1;
                }
                let (unclipped, capped) = (ratio * advantage, clamped * advantage);
                loss -= unclipped.min(capped) * scale;
                // the gradient flows through the unclipped term only when it is the smaller one
                if unclipped <= capped {
                    ratio * advantage
                } else {
                    0.0
                }
            }
        };
        loss -= entropy_coef * entropy * scale;
        for j in 0..outputs {
            out_grad[i * outputs + j] = -(weight * d_log_p[j] + entropy_coef * d_entropy[j]) * scale;
        }
        log_std_grad -= (weight * d_log_p_std + entropy_coef * d_entropy_std) * scale;
        entropy_sum += entropy;
        kl_sum += batch.old_log_probs[i] - log_p;
    }
    let mut grads = policy.backward(&cache, &out_grad);
    if log_std.is_some() {
        grads.push(log_std_grad);
    }
    PgResult {
        loss,
        grads,
        entropy: entropy_sum * scale,
        kl: kl_sum * scale,
        clip_fraction: clipped as f64 * scale,
    }
}

/// GAE(λ) advantages and return targets for one rollout, in order. `values[t]` = V(s_t), `next_values[t]` =
/// V(s_t+1) (the critic's, or 0 without one); `done[t]`: the game ended after step t (no bootstrap); `cut[t]`: an
/// episode boundary after step t, ended or cut off by the step cap (the advantage chain restarts).
pub fn gae(
    rewards: &[f64],
    values: &[f64],
    next_values: &[f64],
    done: &[bool],
    cut: &[bool],
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut next_advantage = 0.0;
    for t in (0..n).rev() {
        let bootstrap = if done[t] { 0.0 } else { next_values[t] };
        let delta = rewards[t] + gamma * bootstrap - values[t];
        if cut[t] {
            next_advantage = 0.0;
        }
        next_advantage = delta + gamma * lambda * next_advantage;
        advantages[t] = next_advantage;
    }
    let returns = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

/// Scale `grads` so their global L2 norm is at most `max_norm` (0 = off); returns the norm before scaling.
fn clip_norm(grads: &mut [f64], max_norm: f64) -> f64 {
    let norm = libm::sqrt(grads.iter().map(|g| g * g).sum::<f64>());
    if max_norm > 0.0 && norm > max_norm {
        let s = max_norm / norm;
        grads.iter_mut().for_each(|g| *g *= s);
    }
    norm
}

#[derive(Default)]
struct Rollout {
    observations: Vec<f64>,
    next_observations: Vec<f64>,
    actions: Vec<f64>,
    log_probs: Vec<f64>,
    rewards: Vec<f64>,
    done: Vec<bool>,
    cut: Vec<bool>,
}

pub struct PgAgent {
    name: &'static str,
    policy: Mlp,
    /// The Gaussian head's log standard deviation (continuous actions); `None` for a softmax.
    log_std: Option<f64>,
    bounds: Option<(f64, f64)>,
    critic: Option<Mlp>,
    policy_adam: Adam,
    critic_adam: Option<Adam>,
    gamma: f64,
    lambda: f64,
    episodic: bool,
    rollout_steps: usize,
    epochs: usize,
    minibatch_size: usize,
    clip: Option<f64>,
    entropy_coef: f64,
    max_grad_norm: f64,
    normalize_advantages: bool,
    shuffle: Rng,
    rollout: Rollout,
    /// What `act` sampled, for `observe`: the raw action and its log-probability under the acting policy.
    pending: Option<(f64, f64)>,
    updates: u64,
    // this iteration's, for `extras`
    stats: [f64; 6], // policy loss, value loss, entropy, kl, clip fraction, grad norm (sums)
    stat_count: u64,
    last_entropy: f64,
}

impl PgAgent {
    pub const PARAMS: [&'static str; 15] = [
        "hidden",
        "hidden_layers",
        "learning_rate",
        "value_learning_rate",
        "gamma",
        "gae_lambda",
        "rollout_steps",
        "epochs",
        "minibatch_size",
        "clip",
        "entropy_coef",
        "max_grad_norm",
        "normalize_advantages",
        "baseline",
        "init_log_std",
    ];

    /// `algorithm`: `reinforce`, `a2c` or `ppo` -- the defaults differ, every parameter can override them.
    pub fn new(
        algorithm: &str,
        observation_size: usize,
        space: ActionSpace,
        seed: u64,
        params: &Params,
    ) -> Result<PgAgent, String> {
        let name = match algorithm {
            "reinforce" => "reinforce",
            "a2c" => "a2c",
            "ppo" => "ppo",
            other => return Err(format!("{other:?} isn't a policy-gradient algorithm")),
        };
        params.check(name, &Self::PARAMS)?;
        let whole = |key: &str, default: f64, min: f64| -> Result<usize, String> {
            let v = params.get(key, default);
            if v < min || v.fract() != 0.0 {
                return Err(format!("{key} must be a whole number >= {min}, got {v}"));
            }
            Ok(v as usize)
        };
        // (learning rate, lambda, rollout, epochs, minibatch (0 = all), clip (0 = off), normalize, critic)
        let (lr, lambda, rollout, epochs, minibatch, clip, normalize, critic) = match name {
            "reinforce" => (
                1e-3,
                1.0,
                2000.0,
                1.0,
                0.0,
                0.0,
                0.0,
                params.get("baseline", 0.0) != 0.0,
            ),
            "a2c" => (7e-4, 0.95, 128.0, 1.0, 0.0, 0.0, 0.0, true),
            _ => (3e-4, 0.95, 2048.0, 4.0, 64.0, 0.2, 1.0, true),
        };
        if name != "reinforce" && params.get("baseline", 1.0) == 0.0 {
            return Err(format!("{name} always has a critic: `baseline` is REINFORCE's"));
        }
        let hidden = vec![whole("hidden", 64.0, 1.0)?; whole("hidden_layers", 2.0, 1.0)?];
        let (outputs, log_std, bounds) = match space {
            ActionSpace::Discrete(n) => (n, None, None),
            ActionSpace::Continuous { low, high } => (1, Some(params.get("init_log_std", -0.5)), Some((low, high))),
        };
        let mut init = Rng::new(seed, Stream::Init);
        let policy = Mlp::init(
            Shape::mlp(observation_size, &hidden, outputs, Activation::Tanh),
            &mut init,
        );
        let critic = critic.then(|| Mlp::init(Shape::mlp(observation_size, &hidden, 1, Activation::Tanh), &mut init));
        let learning_rate = params.get("learning_rate", lr);
        let value_learning_rate = params.get("value_learning_rate", 1e-3);
        let clip = params.get("clip", clip);
        Ok(PgAgent {
            name,
            policy_adam: Adam::new(policy.params.len() + log_std.is_some() as usize, learning_rate),
            critic_adam: critic.as_ref().map(|c| Adam::new(c.params.len(), value_learning_rate)),
            policy,
            log_std,
            bounds,
            critic,
            gamma: params.get("gamma", 0.99),
            lambda: params.get("gae_lambda", lambda),
            episodic: name == "reinforce",
            rollout_steps: whole("rollout_steps", rollout, 1.0)?,
            epochs: whole("epochs", epochs, 1.0)?,
            minibatch_size: whole("minibatch_size", minibatch, 0.0)?,
            clip: (clip > 0.0).then_some(clip),
            entropy_coef: params.get("entropy_coef", 0.01),
            max_grad_norm: params.get("max_grad_norm", 0.5),
            normalize_advantages: params.get("normalize_advantages", normalize) != 0.0,
            shuffle: Rng::new(seed, Stream::Replay),
            rollout: Rollout::default(),
            pending: None,
            updates: 0,
            stats: [0.0; 6],
            stat_count: 0,
            last_entropy: f64::NAN,
        })
    }

    pub fn policy(&self) -> &Mlp {
        &self.policy
    }

    /// The policy's move probabilities (softmax) on one observation; the Gaussian's (mean, std) for a continuous one.
    fn distribution(&self, observation: &[f64]) -> Vec<f64> {
        let out = self.policy.forward(observation);
        match self.log_std {
            None => log_softmax(&out).iter().map(|&l| libm::exp(l)).collect(),
            Some(ls) => vec![out[0], libm::exp(ls)],
        }
    }

    fn to_action(&self, raw: f64) -> Action {
        match self.bounds {
            None => Action::Discrete(raw as usize),
            Some((low, high)) => Action::Continuous(raw.clamp(low, high)),
        }
    }

    fn values(&self, observations: &[f64], n: usize) -> Vec<f64> {
        match &self.critic {
            Some(critic) => critic.forward_batch(observations, n).outputs().to_vec(),
            None => vec![0.0; n],
        }
    }

    fn update(&mut self) {
        let r = std::mem::take(&mut self.rollout);
        let n = r.rewards.len();
        let d = self.policy.shape.inputs();
        let values = self.values(&r.observations, n);
        let next_values = self.values(&r.next_observations, n);
        let (mut advantages, returns) = gae(
            &r.rewards,
            &values,
            &next_values,
            &r.done,
            &r.cut,
            self.gamma,
            self.lambda,
        );
        if self.normalize_advantages && n > 1 {
            let mean = advantages.iter().sum::<f64>() / n as f64;
            let sd = libm::sqrt(advantages.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / n as f64);
            advantages.iter_mut().for_each(|a| *a = (*a - mean) / (sd + 1e-8));
        }
        let size = if self.minibatch_size == 0 {
            n
        } else {
            self.minibatch_size.min(n)
        };
        let mut order: Vec<usize> = (0..n).collect();
        for _ in 0..self.epochs {
            if size < n {
                for i in (1..n).rev() {
                    let j = self.shuffle.below((i + 1) as u32) as usize;
                    order.swap(i, j);
                }
            }
            for chunk in order.chunks(size) {
                let rows = |data: &[f64], width: usize| -> Vec<f64> {
                    chunk
                        .iter()
                        .flat_map(|&i| data[i * width..(i + 1) * width].iter().copied())
                        .collect()
                };
                let batch = PgBatch {
                    observations: rows(&r.observations, d),
                    actions: rows(&r.actions, 1),
                    old_log_probs: rows(&r.log_probs, 1),
                    advantages: rows(&advantages, 1),
                };
                let mut result = pg_gradients(&self.policy, self.log_std, &batch, self.clip, self.entropy_coef);
                let norm = clip_norm(&mut result.grads, self.max_grad_norm);
                let mut params = std::mem::take(&mut self.policy.params);
                if let Some(ls) = self.log_std {
                    params.push(ls);
                }
                self.policy_adam.step(&mut params, &result.grads);
                if self.log_std.is_some() {
                    self.log_std = params.pop();
                }
                self.policy.params = params;
                let mut value_loss = 0.0;
                if let (Some(critic), Some(adam)) = (&mut self.critic, &mut self.critic_adam) {
                    let m = chunk.len();
                    let cache = critic.forward_batch(&batch.observations, m);
                    let target = rows(&returns, 1);
                    let grad: Vec<f64> = cache
                        .outputs()
                        .iter()
                        .zip(&target)
                        .map(|(v, t)| (v - t) / m as f64)
                        .collect();
                    value_loss = cache
                        .outputs()
                        .iter()
                        .zip(&target)
                        .map(|(v, t)| 0.5 * (v - t) * (v - t))
                        .sum::<f64>()
                        / m as f64;
                    let mut grads = critic.backward(&cache, &grad);
                    clip_norm(&mut grads, self.max_grad_norm);
                    adam.step(&mut critic.params, &grads);
                }
                for (slot, v) in self.stats.iter_mut().zip([
                    result.loss,
                    value_loss,
                    result.entropy,
                    result.kl,
                    result.clip_fraction,
                    norm,
                ]) {
                    *slot += v;
                }
                self.stat_count += 1;
                self.last_entropy = result.entropy;
                self.updates += 1;
            }
        }
    }
}

impl Agent for PgAgent {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Sample from the policy (exploration is the policy's own randomness).
    fn act(&mut self, observation: &[f64], rng: &mut Rng) -> Action {
        let out = self.policy.forward(observation);
        let (raw, log_p) = match self.log_std {
            None => {
                let logp = log_softmax(&out);
                let u = rng.uniform();
                let (mut total, mut chosen) = (0.0, logp.len() - 1);
                for (j, &l) in logp.iter().enumerate() {
                    total += libm::exp(l);
                    if u < total {
                        chosen = j;
                        break;
                    }
                }
                (chosen as f64, logp[chosen])
            }
            Some(ls) => {
                let z = rng.normal();
                (out[0] + libm::exp(ls) * z, -0.5 * z * z - ls - HALF_LN_2PI)
            }
        };
        self.pending = Some((raw, log_p));
        self.to_action(raw)
    }

    /// The most likely move (discrete) or the Gaussian's mean: what evaluation plays.
    fn act_greedy(&mut self, observation: &[f64], _rng: &mut Rng) -> Action {
        let out = self.policy.forward(observation);
        match self.log_std {
            None => {
                let mut best = 0;
                for (i, &v) in out.iter().enumerate() {
                    if v > out[best] {
                        best = i;
                    }
                }
                Action::Discrete(best)
            }
            Some(_) => self.to_action(out[0]),
        }
    }

    fn observe(&mut self, t: &Transition, _rng: &mut Rng) {
        let (raw, log_p) = self.pending.take().expect("observe follows act");
        let r = &mut self.rollout;
        r.observations.extend_from_slice(t.observation);
        r.next_observations.extend_from_slice(t.next_observation);
        r.actions.push(raw);
        r.log_probs.push(log_p);
        r.rewards.push(t.reward);
        r.done.push(t.done);
        let boundary = t.done || t.truncated;
        r.cut.push(boundary);
        let full = r.rewards.len() >= self.rollout_steps;
        if full && (!self.episodic || boundary) {
            self.update();
        }
    }

    fn entropy(&self) -> f64 {
        self.last_entropy
    }

    /// The move probabilities the policy acts by (softmax); for a Gaussian, its mean and standard deviation.
    fn action_values(&self, observation: &[f64]) -> Option<Vec<f64>> {
        Some(self.distribution(observation))
    }

    fn extras(&mut self) -> Vec<(String, f64)> {
        let k = self.stat_count.max(1) as f64;
        let names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "kl",
            "clip_fraction",
            "grad_norm",
        ];
        let mut out: Vec<(String, f64)> = if self.stat_count == 0 {
            Vec::new()
        } else {
            names
                .iter()
                .zip(self.stats)
                .map(|(n, s)| (n.to_string(), s / k))
                .collect()
        };
        out.push(("updates".into(), self.updates as f64));
        if let Some(ls) = self.log_std {
            out.push(("action_std".into(), libm::exp(ls)));
        }
        self.stats = [0.0; 6];
        self.stat_count = 0;
        out
    }

    /// `{"type": "mlp", "algorithm", layer_sizes, activations, params}` -- the policy network (logits, or the
    /// Gaussian's mean with its `log_std`).
    fn snapshot(&self) -> String {
        let mlp = &self.policy;
        let sizes: Vec<String> = mlp.shape.layer_sizes.iter().map(|s| s.to_string()).collect();
        let activations: Vec<String> = mlp
            .shape
            .activations
            .iter()
            .map(|a| format!("\"{}\"", a.name()))
            .collect();
        let params: Vec<String> = mlp.params.iter().map(|v| format!("{v:?}")).collect();
        let log_std = self
            .log_std
            .map_or(String::new(), |ls| format!(r#", "log_std": {ls:?}"#));
        format!(
            r#"{{"type": "mlp", "algorithm": "{}", "layer_sizes": [{}], "activations": [{}], "params": [{}]{log_std}}}"#,
            self.name,
            sizes.join(", "),
            activations.join(", "),
            params.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Trainer, TrainerConfig};
    use crate::env::{Env, EnvFactory, Step as EnvStep};

    #[test]
    fn gae_is_monte_carlo_at_lambda_one_and_restarts_at_boundaries() {
        let rewards = [1.0, 0.0, 2.0, 1.0, 1.0];
        let zeros = [0.0; 5];
        let done = [false, false, true, false, false];
        let cut = [false, false, true, false, true];
        let (a, r) = gae(&rewards, &zeros, &zeros, &done, &cut, 0.5, 1.0);
        assert_eq!(a, vec![1.0 + 0.0 + 0.25 * 2.0, 0.0 + 0.5 * 2.0, 2.0, 1.0 + 0.5, 1.0]);
        assert_eq!(a, r);
        // a truncated end (cut, not done) bootstraps from its next value; λ = 0 is one-step TD
        let (a, _) = gae(
            &[1.0, 1.0],
            &[0.5, 0.5],
            &[0.5, 4.0],
            &[false, false],
            &[false, true],
            0.5,
            0.0,
        );
        assert_eq!(a, vec![1.0 + 0.25 - 0.5, 1.0 + 2.0 - 0.5]);
    }

    fn batch(n: usize, inputs: usize, discrete: Option<usize>, rng: &mut Rng) -> PgBatch {
        PgBatch {
            observations: (0..n * inputs).map(|_| rng.normal()).collect(),
            actions: (0..n)
                .map(|_| match discrete {
                    Some(k) => rng.below(k as u32) as f64,
                    None => rng.normal(),
                })
                .collect(),
            old_log_probs: (0..n).map(|_| -1.0 + 0.3 * rng.normal()).collect(),
            advantages: (0..n).map(|_| rng.normal()).collect(),
        }
    }

    #[test]
    fn pg_gradients_match_finite_differences() {
        for (discrete, clip) in [(Some(3), None), (Some(3), Some(0.2)), (None, None), (None, Some(0.2))] {
            let outputs = discrete.unwrap_or(1);
            let policy = Mlp::init(
                Shape::mlp(4, &[5], outputs, Activation::Tanh),
                &mut Rng::new(1, Stream::Init),
            );
            let log_std = discrete.is_none().then_some(-0.3);
            let b = batch(7, 4, discrete, &mut Rng::new(2, Stream::Data));
            let result = pg_gradients(&policy, log_std, &b, clip, 0.05);
            let loss = |params: &[f64]| {
                let (net, ls) = match log_std {
                    Some(_) => (
                        Mlp::new(policy.shape.clone(), params[..params.len() - 1].to_vec()).unwrap(),
                        Some(*params.last().unwrap()),
                    ),
                    None => (Mlp::new(policy.shape.clone(), params.to_vec()).unwrap(), None),
                };
                pg_gradients(&net, ls, &b, clip, 0.05).loss
            };
            let mut params = policy.params.clone();
            params.extend(log_std);
            let h = 1e-6;
            for i in 0..params.len() {
                let original = params[i];
                params[i] = original + h;
                let up = loss(&params);
                params[i] = original - h;
                let down = loss(&params);
                params[i] = original;
                let numeric = (up - down) / (2.0 * h);
                assert!(
                    (numeric - result.grads[i]).abs() < 1e-6,
                    "{discrete:?} {clip:?} param {i}: {numeric} vs {}",
                    result.grads[i]
                );
            }
        }
    }

    /// A corridor observed as a one-hot position: moving right is optimal (tabular.rs and dqn.rs have its cousins).
    const LEN: usize = 6;
    struct Corridor {
        at: usize,
    }
    impl Env for Corridor {
        fn observation_size(&self) -> usize {
            LEN
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
                reward: if done { 1.0 } else { -0.01 },
                done,
            }
        }
        fn score(&self) -> f64 {
            self.at as f64
        }
    }
    impl Corridor {
        fn obs(&self) -> Vec<f64> {
            (0..LEN).map(|i| if i == self.at { 1.0 } else { 0.0 }).collect()
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

    #[test]
    fn every_algorithm_learns_the_corridor() {
        for (algorithm, extra) in [
            ("reinforce", vec![("rollout_steps", 200.0), ("learning_rate", 1e-2)]),
            (
                "reinforce",
                vec![("rollout_steps", 200.0), ("learning_rate", 1e-2), ("baseline", 1.0)],
            ),
            ("a2c", vec![("learning_rate", 3e-3)]),
            ("ppo", vec![("rollout_steps", 256.0), ("learning_rate", 3e-3)]),
        ] {
            let mut params: Vec<(String, f64)> = vec![("hidden".into(), 16.0)];
            params.extend(extra.iter().map(|&(k, v)| (k.to_string(), v)));
            let config = TrainerConfig {
                seed: 3,
                seed_pool: (0, 10),
                max_episode_steps: 50,
            };
            let mut trainer =
                Trainer::build(Box::new(CorridorFactory), algorithm, &Params::new(params), config).unwrap();
            trainer.train(8000);
            let episode = trainer.evaluate(&[0], 20)[0];
            assert_eq!(
                episode.steps as usize,
                LEN - 1,
                "{algorithm} {extra:?} walks straight to the end"
            );
            let probs = trainer.agent().action_values(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
            assert!(
                probs[1] > 0.8 && (probs.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                "{algorithm}: {probs:?}"
            );
        }
    }

    /// One step, one number in: reward -(a - 0.5)^2. A Gaussian policy must move its mean to 0.5.
    struct Aim;
    impl Env for Aim {
        fn observation_size(&self) -> usize {
            1
        }
        fn action_space(&self) -> ActionSpace {
            ActionSpace::Continuous { low: -1.0, high: 1.0 }
        }
        fn reset(&mut self, _seed: u64) -> Vec<f64> {
            vec![1.0]
        }
        fn step(&mut self, action: Action) -> EnvStep {
            let Action::Continuous(a) = action else {
                panic!("continuous")
            };
            EnvStep {
                observation: vec![1.0],
                reward: -(a - 0.5) * (a - 0.5),
                done: true,
            }
        }
        fn score(&self) -> f64 {
            0.0
        }
    }
    struct AimFactory;
    impl EnvFactory for AimFactory {
        fn make(&self) -> Box<dyn Env> {
            Box::new(Aim)
        }
        fn id(&self) -> String {
            "aim".into()
        }
    }

    #[test]
    fn a_gaussian_policy_finds_the_best_action() {
        // Plain REINFORCE heads there too, but noisily (every reward here is negative -- the baseline's whole point),
        // so it is tested with its baseline. Budgets give each the same number of gradient steps (~380): PPO takes 8
        // per rollout (4 epochs x 2 minibatches), the others one.
        for (algorithm, steps) in [("reinforce", 48_000), ("a2c", 48_000), ("ppo", 6_000)] {
            let params = Params::new(
                [
                    ("hidden", 8.0),
                    ("rollout_steps", 128.0),
                    ("learning_rate", 1e-2),
                    ("entropy_coef", 0.0),
                    ("baseline", 1.0),
                ]
                .map(|(k, v)| (k.to_string(), v)),
            );
            let config = TrainerConfig {
                seed: 1,
                seed_pool: (0, 10),
                max_episode_steps: 1,
            };
            let mut trainer = Trainer::build(Box::new(AimFactory), algorithm, &params, config).unwrap();
            trainer.train(steps);
            let dist = trainer.agent().action_values(&[1.0]).unwrap();
            assert!((dist[0] - 0.5).abs() < 0.1, "{algorithm}: mean {}", dist[0]);
            assert_eq!(trainer.act_greedy(&[1.0]), Action::Continuous(dist[0]));
        }
    }

    #[test]
    fn continuous_snapshots_carry_log_std_and_bad_params_are_rejected() {
        let params = Params::new(
            [("hidden", 16.0), ("rollout_steps", 512.0), ("learning_rate", 3e-3)].map(|(k, v)| (k.to_string(), v)),
        );
        let space = ActionSpace::Continuous { low: -1.0, high: 1.0 };
        let agent = PgAgent::new("ppo", 2, space, 0, &params).unwrap();
        assert!(agent.snapshot().contains(r#""log_std": -0.5"#));
        let bad =
            |alg: &str, k: &str, v: f64| PgAgent::new(alg, 2, space, 0, &Params::new([(k.to_string(), v)])).is_err();
        assert!(
            bad("ppo", "baseline", 0.0)
                && bad("ppo", "epochs", 0.0)
                && bad("a2c", "alpha", 0.1)
                && bad("sarsa", "hidden", 4.0)
        );
    }
}
