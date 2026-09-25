//! Deep Q-networks (docs/design/0010 Phase 2): Q-learning with a neural network in place of the table.
//!
//! The network maps an observation to one value per action, so situations that look alike share what is learned --
//! the thing a table can't do. Every piece that makes this stable is a parameter, so each can be switched off and
//! measured as its own experiment arm:
//!
//! - **Experience replay** (`replay_capacity`; 0 = none). Transitions go into a ring buffer and updates train on
//!   random minibatches from it. Without it, each update trains on the last `train_every` transitions, in order:
//!   every transition is used once, and consecutive ones are highly correlated.
//! - **A target network** (`target_update` env steps between copies; 0 = none). Targets are computed by a frozen
//!   copy of the network, so an update doesn't move the target it's chasing.
//! - **Double DQN** (`double`): the online network chooses the next action, the target network values it --
//!   `max` over noisy estimates is biased upwards, and decoupling the two removes most of that bias.
//! - **Dueling heads** (`dueling`): a shared trunk, then a state value `V(s)` and per-action advantages `A(s,a)`,
//!   combined as `Q = V + A - mean(A)`. Since that combination is linear in the trunk's output, a snapshot *folds*
//!   the two heads into one ordinary output layer: every DQN champion is a plain MLP.
//! - **n-step returns** (`n_step`), accumulated exactly like the table's (`tabular.rs`), then stored as one
//!   transition `(s, a, r_0 + g*r_1 + ... , g^n, s_n)`; a game that ended stores discount 0.
//! - **Prioritized replay** (`prioritized`): sample transitions in proportion to `(|td error| + eps)^alpha`
//!   through a sum tree, and correct the bias with importance weights `(N * P(i))^-beta`, beta annealed to 1.
//!
//! The loss is Huber (quadratic within 1 of the target, linear beyond), averaged over the batch; the optimizer is
//! Adam. `td_gradients` is the whole update rule as a pure function of the networks and a batch, which is what
//! `libs/rl/tests/reference_dqn.py` checks against `libs/autodiff`.

use std::collections::VecDeque;

use crate::agent::{Agent, EpsilonSchedule, Params, Transition};
use crate::env::{Action, ActionSpace};
use crate::nn::{Activation, Adam, Cache, Mlp, Shape};
use crate::rng::{Rng, Stream};

/// A Q-network: plain (one MLP, linear outputs) or dueling (a shared ReLU trunk, then value and advantage heads).
#[derive(Clone, Debug)]
pub enum QNet {
    Plain(Mlp),
    Dueling { trunk: Mlp, value: Mlp, advantage: Mlp },
}

/// What a forward pass keeps for the backward pass.
pub enum QCache {
    Plain(Cache),
    Dueling {
        trunk: Cache,
        value: Cache,
        advantage: Cache,
    },
}

impl QNet {
    /// He-uniform weights, zero biases; a dueling net initializes trunk, value head, advantage head in that order.
    pub fn init(inputs: usize, hidden: &[usize], actions: usize, dueling: bool, rng: &mut Rng) -> QNet {
        if !dueling {
            return QNet::Plain(Mlp::init(Shape::mlp(inputs, hidden, actions, Activation::Relu), rng));
        }
        let mut sizes = vec![inputs];
        sizes.extend_from_slice(hidden);
        let last = *sizes.last().unwrap();
        let trunk = Shape::new(sizes, vec![Activation::Relu; hidden.len()]).expect("a trunk with hidden layers");
        QNet::Dueling {
            trunk: Mlp::init(trunk, rng),
            value: Mlp::init(Shape::new(vec![last, 1], vec![Activation::Linear]).unwrap(), rng),
            advantage: Mlp::init(Shape::new(vec![last, actions], vec![Activation::Linear]).unwrap(), rng),
        }
    }

    /// Rebuilds a network from `parts()`'s parameter vectors -- the shape is `like`'s.
    pub fn with_params(like: &QNet, params: &[Vec<f64>]) -> Result<QNet, String> {
        Ok(match like {
            QNet::Plain(mlp) => QNet::Plain(Mlp::new(mlp.shape.clone(), params[0].clone())?),
            QNet::Dueling {
                trunk,
                value,
                advantage,
            } => QNet::Dueling {
                trunk: Mlp::new(trunk.shape.clone(), params[0].clone())?,
                value: Mlp::new(value.shape.clone(), params[1].clone())?,
                advantage: Mlp::new(advantage.shape.clone(), params[2].clone())?,
            },
        })
    }

    pub fn inputs(&self) -> usize {
        match self {
            QNet::Plain(mlp) => mlp.shape.inputs(),
            QNet::Dueling { trunk, .. } => trunk.shape.inputs(),
        }
    }

    pub fn actions(&self) -> usize {
        match self {
            QNet::Plain(mlp) => mlp.shape.outputs(),
            QNet::Dueling { advantage, .. } => advantage.shape.outputs(),
        }
    }

    /// The parameter vectors, one per sub-network (plain: 1; dueling: trunk, value, advantage).
    pub fn parts(&self) -> Vec<&Vec<f64>> {
        match self {
            QNet::Plain(mlp) => vec![&mlp.params],
            QNet::Dueling {
                trunk,
                value,
                advantage,
            } => vec![&trunk.params, &value.params, &advantage.params],
        }
    }

    fn parts_mut(&mut self) -> Vec<&mut Vec<f64>> {
        match self {
            QNet::Plain(mlp) => vec![&mut mlp.params],
            QNet::Dueling {
                trunk,
                value,
                advantage,
            } => vec![&mut trunk.params, &mut value.params, &mut advantage.params],
        }
    }

    /// `batch` rows of Q-values (`actions()` each), and the cache for `backward`.
    pub fn forward(&self, inputs: &[f64], batch: usize) -> (Vec<f64>, QCache) {
        match self {
            QNet::Plain(mlp) => {
                let cache = mlp.forward_batch(inputs, batch);
                (cache.outputs().to_vec(), QCache::Plain(cache))
            }
            QNet::Dueling {
                trunk,
                value,
                advantage,
            } => {
                let t = trunk.forward_batch(inputs, batch);
                let v = value.forward_batch(t.outputs(), batch);
                let a = advantage.forward_batch(t.outputs(), batch);
                let n = self.actions();
                let mut q = vec![0.0; batch * n];
                for row in 0..batch {
                    let adv = &a.outputs()[row * n..(row + 1) * n];
                    let mean = adv.iter().sum::<f64>() / n as f64;
                    for j in 0..n {
                        q[row * n + j] = v.outputs()[row] + adv[j] - mean;
                    }
                }
                (
                    q,
                    QCache::Dueling {
                        trunk: t,
                        value: v,
                        advantage: a,
                    },
                )
            }
        }
    }

    /// Q-values for one observation.
    pub fn q(&self, observation: &[f64]) -> Vec<f64> {
        self.forward(observation, 1).0
    }

    /// Parameter gradients (one vector per `parts()` entry) from the loss's gradient w.r.t. the Q-values.
    pub fn backward(&self, cache: &QCache, q_grad: &[f64]) -> Vec<Vec<f64>> {
        match (self, cache) {
            (QNet::Plain(mlp), QCache::Plain(c)) => vec![mlp.backward(c, q_grad)],
            (
                QNet::Dueling {
                    trunk,
                    value,
                    advantage,
                },
                QCache::Dueling {
                    trunk: tc,
                    value: vc,
                    advantage: ac,
                },
            ) => {
                // q_j = v + a_j - mean(a): dL/dv = sum_j dq_j, dL/da_k = dq_k - mean_j(dq_j)
                let n = self.actions();
                let batch = q_grad.len() / n;
                let mut dv = vec![0.0; batch];
                let mut da = vec![0.0; batch * n];
                for row in 0..batch {
                    let dq = &q_grad[row * n..(row + 1) * n];
                    let total = dq.iter().sum::<f64>();
                    dv[row] = total;
                    for k in 0..n {
                        da[row * n + k] = dq[k] - total / n as f64;
                    }
                }
                let (gv, mut dh) = value.backward_with_inputs(vc, &dv);
                let (ga, dh_a) = advantage.backward_with_inputs(ac, &da);
                for (x, y) in dh.iter_mut().zip(&dh_a) {
                    *x += y;
                }
                vec![trunk.backward(tc, &dh), gv, ga]
            }
            _ => panic!("a cache from a different kind of network"),
        }
    }

    /// The same function as one plain MLP: a dueling net's heads fold into a single linear output layer, since
    /// `V + A_j - mean(A)` is linear in the trunk's output -- row j is `w_v + w_a[j] - mean(w_a)`, likewise the bias.
    pub fn to_mlp(&self) -> Mlp {
        match self {
            QNet::Plain(mlp) => mlp.clone(),
            QNet::Dueling {
                trunk,
                value,
                advantage,
            } => {
                let (h, n) = (value.shape.inputs(), self.actions());
                let (wv, bv) = (&value.params[..h], value.params[h]);
                let (wa, ba) = advantage.params.split_at(n * h);
                let mut params = trunk.params.clone();
                for j in 0..n {
                    for i in 0..h {
                        let mean = (0..n).map(|k| wa[k * h + i]).sum::<f64>() / n as f64;
                        params.push(wv[i] + wa[j * h + i] - mean);
                    }
                }
                let mean_b = ba.iter().sum::<f64>() / n as f64;
                params.extend((0..n).map(|j| bv + ba[j] - mean_b));
                let mut sizes = trunk.shape.layer_sizes.clone();
                sizes.push(n);
                let mut activations = trunk.shape.activations.clone();
                activations.push(Activation::Linear);
                Mlp::new(Shape::new(sizes, activations).unwrap(), params).expect("folded sizes add up")
            }
        }
    }
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// A minibatch of stored transitions: `returns[i]` is the (n-step) discounted reward, `discounts[i]` the factor on
/// the bootstrap from `next_observations[i]` (0 when the game ended), `weights[i]` its importance weight.
#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub observations: Vec<f64>,
    pub actions: Vec<usize>,
    pub returns: Vec<f64>,
    pub discounts: Vec<f64>,
    pub next_observations: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Batch {
    /// Transitions as they came, each weighted 1.
    fn from_stored(stored: &[Stored]) -> Batch {
        Batch {
            observations: stored.iter().flat_map(|t| t.observation.iter().copied()).collect(),
            actions: stored.iter().map(|t| t.action).collect(),
            returns: stored.iter().map(|t| t.ret).collect(),
            discounts: stored.iter().map(|t| t.discount).collect(),
            next_observations: stored.iter().flat_map(|t| t.next_observation.iter().copied()).collect(),
            weights: vec![1.0; stored.len()],
        }
    }
}

pub struct TdResult {
    /// Mean importance-weighted Huber loss.
    pub loss: f64,
    pub grads: Vec<Vec<f64>>,
    /// `target - Q(s, a)` per transition (what prioritized replay sets priorities from).
    pub td: Vec<f64>,
    /// Mean `Q(s, a)` over the batch: the estimates' level (overestimation shows up here).
    pub q_mean: f64,
}

/// One DQN update's loss and gradients: targets `r + discount * Q_target(s', a*)`, with `a*` the target network's
/// argmax (or, `double`, the online network's), and a Huber loss (delta 1) on `target - Q_online(s, a)`.
pub fn td_gradients(online: &QNet, target: &QNet, batch: &Batch, double: bool) -> TdResult {
    let n = online.actions();
    let size = batch.actions.len();
    let (next_target, _) = target.forward(&batch.next_observations, size);
    let next_online = if double {
        Some(online.forward(&batch.next_observations, size).0)
    } else {
        None
    };
    let (q, cache) = online.forward(&batch.observations, size);
    let mut q_grad = vec![0.0; size * n];
    let (mut loss, mut q_sum) = (0.0, 0.0);
    let mut td = Vec::with_capacity(size);
    for i in 0..size {
        let row = &next_target[i * n..(i + 1) * n];
        let chosen = match &next_online {
            Some(next) => argmax(&next[i * n..(i + 1) * n]),
            None => argmax(row),
        };
        let y = batch.returns[i] + batch.discounts[i] * row[chosen];
        let predicted = q[i * n + batch.actions[i]];
        let delta = y - predicted;
        let w = batch.weights[i];
        loss += w * if delta.abs() <= 1.0 {
            0.5 * delta * delta
        } else {
            delta.abs() - 0.5
        };
        q_grad[i * n + batch.actions[i]] = -w * delta.clamp(-1.0, 1.0) / size as f64;
        q_sum += predicted;
        td.push(delta);
    }
    TdResult {
        loss: loss / size as f64,
        grads: online.backward(&cache, &q_grad),
        td,
        q_mean: q_sum / size as f64,
    }
}

/// A sum tree over priorities: leaf `i` holds transition `i`'s priority, each parent the sum of its children, so
/// sampling in proportion to priority is a walk down from the root. Parents are recomputed, never adjusted by a
/// difference, so rounding can't drift -- and every target computes the same sums.
struct SumTree {
    leaves: usize,
    nodes: Vec<f64>,
}

impl SumTree {
    fn new(capacity: usize) -> SumTree {
        let leaves = capacity.next_power_of_two();
        SumTree {
            leaves,
            nodes: vec![0.0; 2 * leaves],
        }
    }

    fn total(&self) -> f64 {
        self.nodes[1]
    }

    fn get(&self, i: usize) -> f64 {
        self.nodes[self.leaves + i]
    }

    fn set(&mut self, i: usize, priority: f64) {
        let mut node = self.leaves + i;
        self.nodes[node] = priority;
        while node > 1 {
            node /= 2;
            self.nodes[node] = self.nodes[2 * node] + self.nodes[2 * node + 1];
        }
    }

    /// The leaf where the running sum passes `mass`, clamped to the filled leaves.
    fn find(&self, mut mass: f64, filled: usize) -> usize {
        let mut node = 1;
        while node < self.leaves {
            let left = self.nodes[2 * node];
            if mass < left {
                node *= 2;
            } else {
                mass -= left;
                node = 2 * node + 1;
            }
        }
        (node - self.leaves).min(filled - 1)
    }
}

/// A ring buffer of transitions, optionally prioritized.
struct Replay {
    capacity: usize,
    observation_size: usize,
    observations: Vec<f64>,
    next_observations: Vec<f64>,
    actions: Vec<usize>,
    returns: Vec<f64>,
    discounts: Vec<f64>,
    len: usize,
    next: usize,
    tree: Option<SumTree>,
    max_priority: f64,
}

impl Replay {
    fn new(capacity: usize, observation_size: usize, prioritized: bool) -> Replay {
        Replay {
            capacity,
            observation_size,
            observations: vec![0.0; capacity * observation_size],
            next_observations: vec![0.0; capacity * observation_size],
            actions: vec![0; capacity],
            returns: vec![0.0; capacity],
            discounts: vec![0.0; capacity],
            len: 0,
            next: 0,
            tree: prioritized.then(|| SumTree::new(capacity)),
            max_priority: 1.0,
        }
    }

    fn push(&mut self, t: &Stored) {
        let (i, d) = (self.next, self.observation_size);
        self.observations[i * d..(i + 1) * d].copy_from_slice(&t.observation);
        self.next_observations[i * d..(i + 1) * d].copy_from_slice(&t.next_observation);
        self.actions[i] = t.action;
        self.returns[i] = t.ret;
        self.discounts[i] = t.discount;
        if let Some(tree) = &mut self.tree {
            tree.set(i, self.max_priority); // new transitions are sampled at least once, soon
        }
        self.next = (self.next + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Indices and importance weights: uniform (weights 1), or stratified proportional sampling with weights
    /// `(len * P(i))^-beta`, normalized by the batch's largest.
    fn sample(&self, size: usize, beta: f64, rng: &mut Rng) -> (Vec<usize>, Vec<f64>) {
        match &self.tree {
            None => (
                (0..size).map(|_| rng.below(self.len as u32) as usize).collect(),
                vec![1.0; size],
            ),
            Some(tree) => {
                let total = tree.total();
                let segment = total / size as f64;
                let indices: Vec<usize> = (0..size)
                    .map(|k| tree.find((k as f64 + rng.uniform()) * segment, self.len))
                    .collect();
                let raw: Vec<f64> = indices
                    .iter()
                    .map(|&i| libm::pow(self.len as f64 * tree.get(i) / total, -beta))
                    .collect();
                let max = raw.iter().cloned().fold(0.0, f64::max);
                (indices, raw.iter().map(|w| w / max).collect())
            }
        }
    }

    fn gather(&self, indices: &[usize], weights: Vec<f64>) -> Batch {
        let d = self.observation_size;
        let rows = |data: &[f64]| -> Vec<f64> {
            indices
                .iter()
                .flat_map(|&i| data[i * d..(i + 1) * d].iter().copied())
                .collect()
        };
        Batch {
            observations: rows(&self.observations),
            actions: indices.iter().map(|&i| self.actions[i]).collect(),
            returns: indices.iter().map(|&i| self.returns[i]).collect(),
            discounts: indices.iter().map(|&i| self.discounts[i]).collect(),
            next_observations: rows(&self.next_observations),
            weights,
        }
    }

    fn set_priority(&mut self, i: usize, priority: f64) {
        if let Some(tree) = &mut self.tree {
            tree.set(i, priority);
            self.max_priority = self.max_priority.max(priority);
        }
    }
}

/// A transition ready for learning: an n-step return and the discount on its bootstrap.
struct Stored {
    observation: Vec<f64>,
    action: usize,
    ret: f64,
    discount: f64,
    next_observation: Vec<f64>,
}

pub struct DqnAgent {
    online: QNet,
    target: Option<QNet>,
    adams: Vec<Adam>,
    replay: Option<Replay>,
    /// Without replay: the transitions since the last update, trained on once, in order.
    recent: Vec<Stored>,
    pending: VecDeque<(Vec<f64>, usize, f64)>,
    epsilon: EpsilonSchedule,
    gamma: f64,
    batch_size: usize,
    learn_start: u64,
    train_every: u64,
    target_update: u64,
    double: bool,
    n_step: usize,
    priority_alpha: f64,
    priority_beta: f64,
    priority_beta_steps: f64,
    sample_rng: Rng,
    acted: u64,
    observed: u64,
    updates: u64,
    // this iteration's
    loss_sum: f64,
    q_sum: f64,
    iteration_updates: u64,
}

impl DqnAgent {
    pub const PARAMS: [&'static str; 19] = [
        "hidden",
        "hidden_layers",
        "learning_rate",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "replay_capacity",
        "batch_size",
        "learn_start",
        "train_every",
        "target_update",
        "double",
        "dueling",
        "n_step",
        "prioritized",
        "priority_alpha",
        "priority_beta",
        "priority_beta_steps",
    ];

    pub fn new(observation_size: usize, space: ActionSpace, seed: u64, params: &Params) -> Result<DqnAgent, String> {
        params.check("dqn", &Self::PARAMS)?;
        let actions = match space {
            ActionSpace::Discrete(n) => n,
            ActionSpace::Continuous { .. } => return Err("dqn needs a discrete action space".into()),
        };
        let whole = |name: &str, default: f64, min: f64| -> Result<usize, String> {
            let v = params.get(name, default);
            if v < min || v.fract() != 0.0 {
                return Err(format!("{name} must be a whole number >= {min}, got {v}"));
            }
            Ok(v as usize)
        };
        let flag = |name: &str| params.get(name, 0.0) != 0.0;
        let hidden = vec![whole("hidden", 64.0, 1.0)?; whole("hidden_layers", 2.0, 1.0)?];
        let (gamma, learning_rate) = (params.get("gamma", 0.95), params.get("learning_rate", 5e-4));
        if !((0.0..=1.0).contains(&gamma) && learning_rate > 0.0) {
            return Err(format!(
                "need 0 <= gamma <= 1 and learning_rate > 0, got {gamma}, {learning_rate}"
            ));
        }
        let replay_capacity = whole("replay_capacity", 50_000.0, 0.0)?;
        let batch_size = whole("batch_size", 32.0, 1.0)?;
        let prioritized = flag("prioritized");
        if prioritized && replay_capacity == 0 {
            return Err("prioritized replay needs a replay buffer (replay_capacity > 0)".into());
        }
        let dueling = flag("dueling");
        let online = QNet::init(
            observation_size,
            &hidden,
            actions,
            dueling,
            &mut Rng::new(seed, Stream::Init),
        );
        let target_update = whole("target_update", 2_000.0, 0.0)? as u64;
        Ok(DqnAgent {
            adams: online
                .parts()
                .iter()
                .map(|p| Adam::new(p.len(), learning_rate))
                .collect(),
            target: (target_update > 0).then(|| online.clone()),
            online,
            replay: (replay_capacity > 0).then(|| Replay::new(replay_capacity, observation_size, prioritized)),
            recent: Vec::new(),
            pending: VecDeque::new(),
            epsilon: EpsilonSchedule::from_params(params),
            gamma,
            batch_size,
            learn_start: whole("learn_start", 1_000.0, 0.0)? as u64,
            train_every: whole("train_every", 4.0, 1.0)? as u64,
            target_update,
            double: flag("double"),
            n_step: whole("n_step", 1.0, 1.0)?,
            priority_alpha: params.get("priority_alpha", 0.6),
            priority_beta: params.get("priority_beta", 0.4),
            priority_beta_steps: params.get("priority_beta_steps", 1_000_000.0).max(1.0),
            sample_rng: Rng::new(seed, Stream::Replay),
            acted: 0,
            observed: 0,
            updates: 0,
            loss_sum: 0.0,
            q_sum: 0.0,
            iteration_updates: 0,
        })
    }

    pub fn network(&self) -> &QNet {
        &self.online
    }

    /// Move the oldest pending step into learning, as an n-step transition ending in `next_observation`.
    fn emit_oldest(&mut self, next_observation: &[f64], ended: bool) {
        let (mut ret, mut discount) = (0.0, 1.0);
        for &(_, _, reward) in self.pending.iter().rev() {
            ret = reward + self.gamma * ret;
            discount *= self.gamma;
        }
        let (observation, action, _) = self.pending.pop_front().expect("something pending");
        let stored = Stored {
            observation,
            action,
            ret,
            discount: if ended { 0.0 } else { discount },
            next_observation: next_observation.to_vec(),
        };
        match &mut self.replay {
            Some(replay) => replay.push(&stored),
            None => self.recent.push(stored),
        }
    }

    fn update(&mut self) {
        let beta = self.priority_beta
            + (1.0 - self.priority_beta) * (self.observed as f64 / self.priority_beta_steps).min(1.0);
        // the batch: sampled from replay, or the transitions since the last update, as they came
        let (indices, batch) = match &self.replay {
            Some(replay) if replay.len > 0 => {
                let (indices, weights) = replay.sample(self.batch_size, beta, &mut self.sample_rng);
                let batch = replay.gather(&indices, weights);
                (indices, batch)
            }
            None if !self.recent.is_empty() => (Vec::new(), Batch::from_stored(&std::mem::take(&mut self.recent))),
            _ => return,
        };
        let result = td_gradients(
            &self.online,
            self.target.as_ref().unwrap_or(&self.online),
            &batch,
            self.double,
        );
        for ((params, adam), grads) in self
            .online
            .parts_mut()
            .into_iter()
            .zip(&mut self.adams)
            .zip(&result.grads)
        {
            adam.step(params, grads);
        }
        if let Some(replay) = &mut self.replay {
            for (&i, delta) in indices.iter().zip(&result.td) {
                replay.set_priority(i, libm::pow(delta.abs() + 1e-6, self.priority_alpha));
            }
        }
        self.updates += 1;
        self.iteration_updates += 1;
        self.loss_sum += result.loss;
        self.q_sum += result.q_mean;
    }
}

impl Agent for DqnAgent {
    fn name(&self) -> &'static str {
        "dqn"
    }

    fn act(&mut self, observation: &[f64], rng: &mut Rng) -> Action {
        let eps = self.epsilon.at(self.acted);
        self.acted += 1;
        let n = self.online.actions();
        if rng.uniform() < eps {
            Action::Discrete(rng.below(n as u32) as usize)
        } else {
            Action::Discrete(argmax(&self.online.q(observation)))
        }
    }

    fn act_greedy(&mut self, observation: &[f64], _rng: &mut Rng) -> Action {
        Action::Discrete(argmax(&self.online.q(observation)))
    }

    fn observe(&mut self, t: &Transition, _rng: &mut Rng) {
        let action = match t.action {
            Action::Discrete(a) => a,
            other => panic!("dqn chose {other:?}, not a discrete action"),
        };
        self.pending.push_back((t.observation.to_vec(), action, t.reward));
        if t.done || t.truncated {
            while !self.pending.is_empty() {
                self.emit_oldest(t.next_observation, t.done);
            }
        } else if self.pending.len() == self.n_step {
            self.emit_oldest(t.next_observation, false);
        }
        self.observed += 1;
        if self.observed >= self.learn_start && self.observed.is_multiple_of(self.train_every) {
            self.update();
        } else if self.replay.is_none() && self.observed.is_multiple_of(self.train_every) {
            self.recent.clear(); // before learn_start: nothing to train on yet
        }
        if self.target_update > 0 && self.observed.is_multiple_of(self.target_update) {
            self.target = Some(self.online.clone());
        }
    }

    fn entropy(&self) -> f64 {
        self.epsilon.entropy(self.acted, self.online.actions())
    }

    fn extras(&mut self) -> Vec<(String, f64)> {
        let per = self.iteration_updates.max(1) as f64;
        let mut extras = vec![
            ("epsilon".into(), self.epsilon.at(self.acted)),
            ("updates".into(), self.updates as f64),
            ("td_loss".into(), self.loss_sum / per),
            ("q_mean".into(), self.q_sum / per),
        ];
        if let Some(replay) = &self.replay {
            extras.push(("replay_size".into(), replay.len as f64));
        }
        self.loss_sum = 0.0;
        self.q_sum = 0.0;
        self.iteration_updates = 0;
        extras
    }

    /// `{"type": "mlp", "algorithm": "dqn", layer_sizes, activations, params}` -- dueling heads folded in.
    fn snapshot(&self) -> String {
        let mlp = self.online.to_mlp();
        let sizes: Vec<String> = mlp.shape.layer_sizes.iter().map(|s| s.to_string()).collect();
        let activations: Vec<String> = mlp
            .shape
            .activations
            .iter()
            .map(|a| format!("\"{}\"", a.name()))
            .collect();
        let params: Vec<String> = mlp.params.iter().map(|v| format!("{v:?}")).collect();
        format!(
            r#"{{"type": "mlp", "algorithm": "dqn", "layer_sizes": [{}], "activations": [{}], "params": [{}]}}"#,
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

    fn batch_of(size: usize, inputs: usize, actions: usize, rng: &mut Rng) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let obs = (0..size * inputs).map(|_| rng.normal()).collect();
        let acts = (0..size).map(|_| rng.below(actions as u32) as usize).collect();
        let rets = (0..size).map(|_| rng.normal() * 2.0).collect();
        (obs, acts, rets)
    }

    #[test]
    fn dueling_folds_into_the_same_function() {
        let mut rng = Rng::new(1, Stream::Init);
        let net = QNet::init(5, &[8, 6], 3, true, &mut rng);
        let mlp = net.to_mlp();
        let mut data = Rng::new(2, Stream::Data);
        let x: Vec<f64> = (0..5 * 4).map(|_| data.normal()).collect();
        let (q, _) = net.forward(&x, 4);
        let folded = mlp.forward_batch(&x, 4);
        for (a, b) in q.iter().zip(folded.outputs()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn td_gradients_match_finite_differences() {
        for (dueling, double) in [(false, false), (true, false), (false, true), (true, true)] {
            let online = QNet::init(4, &[6], 3, dueling, &mut Rng::new(3, Stream::Init));
            let target = QNet::init(4, &[6], 3, dueling, &mut Rng::new(4, Stream::Init));
            let mut data = Rng::new(5, Stream::Data);
            let (obs, actions, returns) = batch_of(6, 4, 3, &mut data);
            let (next, _, _) = batch_of(6, 4, 3, &mut data);
            let batch = Batch {
                observations: obs,
                actions,
                returns,
                discounts: vec![0.9, 0.0, 0.81, 0.9, 0.9, 0.0],
                next_observations: next,
                weights: vec![1.0, 0.5, 0.25, 1.0, 0.8, 1.0],
            };
            let result = td_gradients(&online, &target, &batch, double);
            let h = 1e-6;
            for (part, grads) in result.grads.iter().enumerate() {
                for (i, &g) in grads.iter().enumerate() {
                    // targets come from the separate target net, and double DQN's argmax is piecewise constant, so
                    // the loss as a function of the online parameters is just the update's own loss
                    let mut params: Vec<Vec<f64>> = online.parts().into_iter().cloned().collect();
                    params[part][i] += h;
                    let up = QNet::with_params(&online, &params).unwrap();
                    params[part][i] -= 2.0 * h;
                    let down = QNet::with_params(&online, &params).unwrap();
                    let loss = |net: &QNet| td_gradients(net, &target, &batch, double).loss;
                    let numeric = (loss(&up) - loss(&down)) / (2.0 * h);
                    assert!(
                        (numeric - g).abs() < 1e-6,
                        "dueling {dueling} double {double} part {part} param {i}: {numeric} vs {g}"
                    );
                }
            }
        }
    }

    #[test]
    fn sum_tree_samples_in_proportion() {
        let mut tree = SumTree::new(5);
        for (i, p) in [1.0, 0.0, 3.0, 2.0, 4.0].iter().enumerate() {
            tree.set(i, *p);
        }
        assert_eq!(tree.total(), 10.0);
        assert_eq!(tree.find(0.5, 5), 0);
        assert_eq!(tree.find(1.0, 5), 2, "a zero-priority leaf is never found");
        assert_eq!(tree.find(3.99, 5), 2);
        assert_eq!(tree.find(4.0, 5), 3);
        assert_eq!(tree.find(9.99, 5), 4);
        tree.set(4, 0.5);
        assert_eq!(tree.total(), 6.5);
    }

    /// A corridor like tabular.rs's, observed as a one-hot position: always moving right is optimal.
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

    fn train(extra: &[(&str, f64)]) -> Trainer {
        let mut params: Vec<(String, f64)> = vec![
            ("epsilon_decay_steps".into(), 2000.0),
            ("hidden".into(), 16.0),
            ("learn_start".into(), 200.0),
            ("target_update".into(), 200.0),
            ("learning_rate".into(), 3e-3),
            ("train_every".into(), 1.0),
        ];
        params.extend(extra.iter().map(|&(k, v)| (k.to_string(), v)));
        let config = TrainerConfig {
            seed: 3,
            seed_pool: (0, 10),
            max_episode_steps: 50,
        };
        let mut trainer = Trainer::build(Box::new(CorridorFactory), "dqn", &Params::new(params), config).unwrap();
        trainer.train(4000);
        trainer
    }

    #[test]
    fn every_variant_learns_the_corridor() {
        for extra in [
            vec![],
            vec![("double", 1.0)],
            vec![("dueling", 1.0)],
            vec![("n_step", 3.0)],
            vec![("prioritized", 1.0)],
            vec![("double", 1.0), ("dueling", 1.0), ("n_step", 3.0), ("prioritized", 1.0)],
        ] {
            let mut trainer = train(&extra);
            let episode = trainer.evaluate(&[0], 20)[0];
            assert_eq!(episode.steps as usize, LEN - 1, "{extra:?} walks straight to the end");
        }
    }

    #[test]
    fn snapshots_are_plain_mlps_and_bad_params_are_rejected() {
        let trainer = train(&[("dueling", 1.0)]);
        let json = trainer.agent().snapshot();
        assert!(json.starts_with(r#"{"type": "mlp", "algorithm": "dqn", "layer_sizes": [6, 16, 16, 2]"#));
        assert!(json.contains(r#""activations": ["relu", "relu", "linear"]"#));
        let space = ActionSpace::Discrete(2);
        let bad = |k: &str, v: f64| DqnAgent::new(4, space, 0, &Params::new([(k.to_string(), v)])).is_err();
        assert!(bad("hidden", 0.0) && bad("n_step", 1.5) && bad("replay_capacty", 1.0));
        let per_without_replay = Params::new([("prioritized".to_string(), 1.0), ("replay_capacity".to_string(), 0.0)]);
        assert!(DqnAgent::new(4, space, 0, &per_without_replay).is_err());
        assert!(DqnAgent::new(
            4,
            ActionSpace::Continuous { low: -1.0, high: 1.0 },
            0,
            &Params::default()
        )
        .is_err());
    }

    #[test]
    fn without_replay_each_update_trains_on_the_recent_steps_once() {
        let mut agent = DqnAgent::new(
            2,
            ActionSpace::Discrete(2),
            0,
            &Params::new([
                ("replay_capacity".to_string(), 0.0),
                ("learn_start".to_string(), 0.0),
                ("train_every".to_string(), 3.0),
            ]),
        )
        .unwrap();
        let mut rng = Rng::new(0, Stream::Explore);
        let (a, b) = ([1.0, 0.0], [0.0, 1.0]);
        for i in 0..6 {
            agent.observe(
                &Transition {
                    observation: &a,
                    action: Action::Discrete(i % 2),
                    reward: 0.1,
                    next_observation: &b,
                    done: false,
                    truncated: false,
                },
                &mut rng,
            );
            assert_eq!(agent.recent.len(), (i + 1) % 3, "staged until the update uses them");
        }
        assert_eq!(agent.updates, 2);
    }
}
