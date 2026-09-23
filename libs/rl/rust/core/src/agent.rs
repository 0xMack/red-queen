//! Agents, and the loop that trains one (docs/design/0010 Decision 3).
//!
//! An `Agent` chooses actions and (if it learns) updates itself from transitions; a `Trainer` owns one agent and one
//! environment and advances training by a budget of *environment steps* -- the unit every RL run is measured in,
//! and one "iteration" of an RL run's telemetry. Phase 0 has one agent, `RandomAgent`: it learns nothing, which is
//! exactly what a pipeline test needs.

use std::collections::BTreeMap;

use crate::env::{play, Action, ActionSpace, Env, EnvFactory, Episode};
use crate::rng::{Rng, Stream};

/// One transition, for an agent that learns from them.
pub struct Transition<'a> {
    pub observation: &'a [f64],
    pub action: Action,
    pub reward: f64,
    pub next_observation: &'a [f64],
    /// The game ended (not merely the step cap): nothing follows `next_observation`.
    pub done: bool,
}

pub trait Agent {
    /// The algorithm's id, recorded as a run's `representation` (`random`, `q_learning`, ...).
    fn name(&self) -> &'static str;
    /// Choose an action while training -- exploring, if the algorithm explores.
    fn act(&mut self, observation: &[f64], rng: &mut Rng) -> Action;
    /// The action the agent is judged by: no exploration. `rng` is for agents that are random by nature.
    fn act_greedy(&mut self, observation: &[f64], rng: &mut Rng) -> Action;
    /// Learn from one transition. The default learns nothing.
    fn observe(&mut self, _transition: &Transition) {}
    /// Mean entropy (nats) of the policy it currently acts by -- a run's `diversity`.
    fn entropy(&self) -> f64;
    /// Algorithm-specific numbers for this iteration (`epsilon`, `td_loss`, ...), a run's `extras`.
    fn extras(&self) -> Vec<(String, f64)> {
        Vec::new()
    }
    /// The current policy in its wire format (JSON): a run's champion artifact.
    fn snapshot(&self) -> String;
}

/// Hyperparameters by name: what a job passes in. Unknown names are an error, not silently ignored.
#[derive(Clone, Debug, Default)]
pub struct Params {
    values: BTreeMap<String, f64>,
}

impl Params {
    pub fn new(values: impl IntoIterator<Item = (String, f64)>) -> Params {
        Params {
            values: values.into_iter().collect(),
        }
    }

    pub fn get(&self, name: &str, default: f64) -> f64 {
        self.values.get(name).copied().unwrap_or(default)
    }

    /// Fails on any name not in `known`: a typo'd hyperparameter must not train with the default.
    pub fn check(&self, algorithm: &str, known: &[&str]) -> Result<(), String> {
        match self.values.keys().find(|k| !known.contains(&k.as_str())) {
            Some(unknown) => Err(format!("{algorithm} has no parameter {unknown:?} (known: {known:?})")),
            None => Ok(()),
        }
    }
}

/// Acts uniformly at random and learns nothing: the floor every learner must beat, and the pipeline's smoke test.
pub struct RandomAgent {
    space: ActionSpace,
}

impl RandomAgent {
    pub fn new(space: ActionSpace) -> RandomAgent {
        RandomAgent { space }
    }

    fn sample(&self, rng: &mut Rng) -> Action {
        match self.space {
            ActionSpace::Discrete(n) => Action::Discrete(rng.below(n as u32) as usize),
            ActionSpace::Continuous { low, high } => Action::Continuous(low + (high - low) * rng.uniform()),
        }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> &'static str {
        "random"
    }

    fn act(&mut self, _observation: &[f64], rng: &mut Rng) -> Action {
        self.sample(rng)
    }

    fn act_greedy(&mut self, _observation: &[f64], rng: &mut Rng) -> Action {
        self.sample(rng)
    }

    fn entropy(&self) -> f64 {
        match self.space {
            ActionSpace::Discrete(n) => libm::log(n as f64),
            // differential entropy of U(low, high)
            ActionSpace::Continuous { low, high } => libm::log(high - low),
        }
    }

    fn snapshot(&self) -> String {
        match self.space {
            ActionSpace::Discrete(n) => format!(r#"{{"type": "random", "num_actions": {n}}}"#),
            ActionSpace::Continuous { low, high } => format!(r#"{{"type": "random", "low": {low}, "high": {high}}}"#),
        }
    }
}

/// Builds an agent by algorithm name.
pub fn build_agent(
    algorithm: &str,
    _observation_size: usize,
    space: ActionSpace,
    params: &Params,
    _seed: u64,
) -> Result<Box<dyn Agent>, String> {
    match algorithm {
        "random" => {
            params.check("random", &[])?;
            Ok(Box::new(RandomAgent::new(space)))
        }
        other => Err(format!("unknown algorithm {other:?} (known: random)")),
    }
}

#[derive(Clone, Debug)]
pub struct TrainerConfig {
    /// The run's seed: every random stream derives from it.
    pub seed: u64,
    /// Training games are drawn from `[lo, hi)` -- disjoint from the leaderboard's and the monitor's seeds.
    pub seed_pool: (u64, u64),
    /// Training episodes are cut after this many steps (the game may end sooner).
    pub max_episode_steps: u32,
}

/// What one call to `Trainer::train` did.
#[derive(Clone, Debug)]
pub struct IterationStats {
    pub steps: u64,
    /// Episodes that ended during this iteration (an episode can span iterations).
    pub episodes: Vec<Episode>,
    pub entropy: f64,
    pub extras: Vec<(String, f64)>,
    pub total_steps: u64,
    pub total_episodes: u64,
}

pub struct Trainer {
    factory: Box<dyn EnvFactory>,
    env: Box<dyn Env>,
    agent: Box<dyn Agent>,
    config: TrainerConfig,
    explore: Rng,
    episode_seeds: Rng,
    // the episode in progress
    observation: Vec<f64>,
    episode_seed: u64,
    episode_reward: f64,
    episode_steps: u32,
    total_steps: u64,
    total_episodes: u64,
}

impl Trainer {
    pub fn new(factory: Box<dyn EnvFactory>, agent: Box<dyn Agent>, config: TrainerConfig) -> Trainer {
        let env = factory.make();
        let mut trainer = Trainer {
            explore: Rng::new(config.seed, Stream::Explore),
            episode_seeds: Rng::new(config.seed, Stream::Episodes),
            factory,
            env,
            agent,
            config,
            observation: Vec::new(),
            episode_seed: 0,
            episode_reward: 0.0,
            episode_steps: 0,
            total_steps: 0,
            total_episodes: 0,
        };
        trainer.begin_episode();
        trainer
    }

    pub fn build(
        factory: Box<dyn EnvFactory>,
        algorithm: &str,
        params: &Params,
        config: TrainerConfig,
    ) -> Result<Trainer, String> {
        let probe = factory.make();
        let agent = build_agent(
            algorithm,
            probe.observation_size(),
            probe.action_space(),
            params,
            config.seed,
        )?;
        Ok(Trainer::new(factory, agent, config))
    }

    fn begin_episode(&mut self) {
        let (lo, hi) = self.config.seed_pool;
        self.episode_seed = self.episode_seeds.range_u64(lo, hi);
        self.observation = self.env.reset(self.episode_seed);
        self.episode_reward = 0.0;
        self.episode_steps = 0;
    }

    pub fn agent(&self) -> &dyn Agent {
        self.agent.as_ref()
    }

    pub fn env_id(&self) -> String {
        self.factory.id()
    }

    /// Advance training by `steps` environment steps.
    pub fn train(&mut self, steps: u64) -> IterationStats {
        let mut finished = Vec::new();
        for _ in 0..steps {
            let action = self.agent.act(&self.observation, &mut self.explore);
            let step = self.env.step(action);
            self.agent.observe(&Transition {
                observation: &self.observation,
                action,
                reward: step.reward,
                next_observation: &step.observation,
                done: step.done,
            });
            self.episode_reward += step.reward;
            self.episode_steps += 1;
            self.total_steps += 1;
            if step.done || self.episode_steps >= self.config.max_episode_steps {
                finished.push(Episode {
                    seed: self.episode_seed,
                    total_reward: self.episode_reward,
                    steps: self.episode_steps,
                    score: self.env.score(),
                });
                self.total_episodes += 1;
                self.begin_episode();
            } else {
                self.observation = step.observation;
            }
        }
        IterationStats {
            steps,
            episodes: finished,
            entropy: self.agent.entropy(),
            extras: self.agent.extras(),
            total_steps: self.total_steps,
            total_episodes: self.total_episodes,
        }
    }

    /// The greedy policy on games `seeds`, each capped at `max_steps` -- in fresh environments, so the episode
    /// being trained on is untouched. Agents that are random by nature draw from a stream seeded by the game.
    pub fn evaluate(&mut self, seeds: &[u64], max_steps: u32) -> Vec<Episode> {
        let mut env = self.factory.make();
        let agent = &mut self.agent;
        seeds
            .iter()
            .map(|&seed| {
                let mut rng = Rng::new(seed, Stream::Explore);
                play(env.as_mut(), seed, max_steps, |obs| agent.act_greedy(obs, &mut rng))
            })
            .collect()
    }
}
