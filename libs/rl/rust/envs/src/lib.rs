//! The games crate's environments as the RL core's `Env` (docs/design/0010 Decision 1) -- integration glue, kept out
//! of both the learning core (which never names a game) and the games core (which never knows about learning), and
//! shared by the PyO3 and WASM bindings so neither duplicates it.
//!
//! - **Snake**: any native observer in, `relative3.v1` out (`Discrete(3)`: 0/1/2 = turn left/straight/right, the
//!   argmax order of a 3-output model), the game's own score. Game `seed` is exactly `games.snake.Snake(seed=...)`.
//!   Reward is the game's shaped one (`Reward::Shaped`) or only its outcomes -- +1 food, -1 death, 0 otherwise
//!   (`Reward::Sparse`) -- so an experiment can ask what the shaping buys a learner.
//! - **Reach1D**: continuous acceleration in `[-1, 1]`; game `seed` draws the target in `[-5, 5)` (start at rest at
//!   0). Never ends on its own -- episodes end at the step cap. Score: minus the final distance to the target.
//!
//! Also here: the games' fixed baselines as policies, so `tests/test_envs.py` can check that an episode through an
//! adapter scores exactly what `jobs/evaluate.py` records for the same seed.

use redqueen_games::baselines::{snake_greedy, SnakeRandom};
use redqueen_games::pcg::Pcg32;
use redqueen_games::reach1d::Reach1D;
use redqueen_games::snake::{Observer, Snake};
use redqueen_rl::agent::{Params, Trainer, TrainerConfig};
use redqueen_rl::digest::Fnv;
use redqueen_rl::env::{play, Action, ActionSpace, Env, EnvFactory, Episode, Step};
use redqueen_rl::tabular::Discretizer;

/// Which reward a Snake learner sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reward {
    /// The game's own: +1 food, -1 death or starvation, +0.01 closer to the food, -0.02 farther.
    Shaped,
    /// Outcomes only: +1 food, -1 death or starvation, 0 for every other step.
    Sparse,
}

impl Reward {
    pub fn parse(name: &str) -> Result<Reward, String> {
        match name {
            "shaped" => Ok(Reward::Shaped),
            "sparse" => Ok(Reward::Sparse),
            other => Err(format!("unknown reward {other:?} (shaped, sparse)")),
        }
    }
}

pub struct SnakeEnv {
    game: Snake,
    observer: Observer,
    reward: Reward,
}

impl Env for SnakeEnv {
    fn observation_size(&self) -> usize {
        self.game.observation_size(self.observer)
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete(3)
    }

    fn reset(&mut self, seed: u64) -> Vec<f64> {
        self.game.seed = seed;
        self.game.reset();
        self.game.encode(self.observer)
    }

    fn step(&mut self, action: Action) -> Step {
        let turn = match action {
            Action::Discrete(i) if i < 3 => i as i64 - 1,
            other => panic!("Snake takes Discrete(0..3), got {other:?}"),
        };
        let (shaped, done) = self.game.step(turn);
        let reward = match self.reward {
            Reward::Shaped => shaped,
            Reward::Sparse if shaped >= 1.0 || shaped <= -1.0 => shaped,
            Reward::Sparse => 0.0,
        };
        Step {
            observation: self.game.encode(self.observer),
            reward,
            done,
        }
    }

    fn score(&self) -> f64 {
        self.game.score as f64
    }

    /// `features.v1` is 11 binary features: 2,048 table rows. The egocentric rays and the full grid are far too
    /// large (or continuous) for a table -- that's what a Q-*network* (Phase 2) is for.
    fn discretizer(&self) -> Option<Discretizer> {
        (self.observer == Observer::Features).then_some(Discretizer::Binary { bits: 11 })
    }
}

pub struct SnakeFactory {
    pub width: i32,
    pub height: i32,
    pub observer: Observer,
    pub max_steps_without_food: Option<u32>,
    pub reward: Reward,
}

impl EnvFactory for SnakeFactory {
    fn make(&self) -> Box<dyn Env> {
        Box::new(SnakeEnv {
            game: Snake::new(self.width, self.height, 0, self.max_steps_without_food),
            observer: self.observer,
            reward: self.reward,
        })
    }

    fn id(&self) -> String {
        format!("snake/{}+relative3.v1", self.observer.id())
    }
}

pub struct Reach1DEnv {
    game: Reach1D,
}

impl Env for Reach1DEnv {
    fn observation_size(&self) -> usize {
        2
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Continuous { low: -1.0, high: 1.0 }
    }

    fn reset(&mut self, seed: u64) -> Vec<f64> {
        let mut rng = Pcg32::new(seed);
        let u = rng.next_u32() as f64 / 4_294_967_296.0;
        self.game.target = -5.0 + 10.0 * u;
        let (offset, velocity) = self.game.reset();
        vec![offset, velocity]
    }

    fn step(&mut self, action: Action) -> Step {
        let acceleration = match action {
            Action::Continuous(a) => a,
            other => panic!("Reach1D takes Continuous, got {other:?}"),
        };
        let ((offset, velocity), reward) = self.game.step(acceleration);
        Step {
            observation: vec![offset, velocity],
            reward,
            done: false,
        }
    }

    fn score(&self) -> f64 {
        -(self.game.position - self.game.target).abs()
    }

    /// Offset from the target in 25 bins over [-6, 6] (targets are within 5 of the start) and velocity in 13 over
    /// [-3, 3] (damping caps speed near 4.9; the end bins take the rest): 325 rows.
    fn discretizer(&self) -> Option<Discretizer> {
        Some(Discretizer::Bins {
            dims: vec![(-6.0, 6.0, 25), (-3.0, 3.0, 13)],
        })
    }
}

pub struct Reach1DFactory;

impl EnvFactory for Reach1DFactory {
    fn make(&self) -> Box<dyn Env> {
        // games.reach1d.ReachTarget1D's defaults
        Box::new(Reach1DEnv {
            game: Reach1D::new(5.0, 0.0, 0.0, 0.1, 1.0, 0.98),
        })
    }

    fn id(&self) -> String {
        "reach1d".into()
    }
}

/// An environment configuration by id: an interface id (`snake/<observer>+relative3.v1`, on a `width` x `height`
/// board) or `reach1d`. Snake's reward is the game's shaped one.
pub fn factory(env_id: &str, width: i32, height: i32) -> Result<Box<dyn EnvFactory>, String> {
    factory_with(env_id, width, height, Reward::Shaped)
}

/// `factory` with a choice of Snake reward (Reach1D has only one).
pub fn factory_with(env_id: &str, width: i32, height: i32, reward: Reward) -> Result<Box<dyn EnvFactory>, String> {
    if env_id == "reach1d" {
        return Ok(Box::new(Reach1DFactory));
    }
    let observer = env_id
        .strip_prefix("snake/")
        .and_then(|rest| rest.strip_suffix("+relative3.v1"))
        .ok_or_else(|| format!("unknown environment {env_id:?} (snake/<observer>+relative3.v1, or reach1d)"))?;
    let observer = Observer::from_id(observer).ok_or_else(|| format!("unknown snake observer {observer:?}"))?;
    if !Snake::fits(width, height) {
        return Err(format!("a {width}x{height} board can't fit the starting snake"));
    }
    Ok(Box::new(SnakeFactory {
        width,
        height,
        observer,
        max_steps_without_food: None,
        reward,
    }))
}

/// A games-crate baseline (`games.baselines`) on each of `seeds`: `snake-random` (a fresh `SnakeRandom(seed)` per
/// game, as `jobs/evaluate.py` builds it) or `snake-greedy`. Both read `features.v1`.
pub fn evaluate_baseline(
    name: &str,
    factory: &dyn EnvFactory,
    seeds: &[u64],
    max_steps: u32,
) -> Result<Vec<Episode>, String> {
    let mut env = factory.make();
    if factory.id() != "snake/features.v1+relative3.v1" {
        return Err(format!("the snake baselines read features.v1, not {}", factory.id()));
    }
    let to_action = |turn: i32| Action::Discrete((turn + 1) as usize);
    seeds
        .iter()
        .map(|&seed| match name {
            "snake-random" => {
                let mut policy = SnakeRandom::new(seed);
                Ok(play(env.as_mut(), seed, max_steps, |obs| to_action(policy.decide(obs))))
            }
            "snake-greedy" => Ok(play(env.as_mut(), seed, max_steps, |obs| to_action(snake_greedy(obs)))),
            other => Err(format!("unknown baseline {other:?} (snake-random, snake-greedy)")),
        })
        .collect()
}

/// The pipeline digest: a random agent trained through `Trainer` on Snake and on Reach1D, then evaluated -- every
/// episode's return, length and score, hashed. With `nn`'s training digest, the checked-in determinism fixture.
pub fn rollout_digest(seed: u64) -> String {
    let mut hash = Fnv::default();
    for env_id in ["snake/features.v1+relative3.v1", "reach1d"] {
        let config = TrainerConfig {
            seed,
            seed_pool: (100_000, 1_000_000),
            max_episode_steps: 200,
        };
        let mut trainer =
            Trainer::build(factory(env_id, 10, 10).unwrap(), "random", &Params::default(), config).unwrap();
        let stats = trainer.train(5_000);
        let evaluated = trainer.evaluate(&[10_000, 10_001, 10_002], 300);
        for episode in stats.episodes.iter().chain(&evaluated) {
            hash.f64(episode.seed as f64);
            hash.f64(episode.total_reward);
            hash.f64(episode.steps as f64);
            hash.f64(episode.score);
        }
    }
    hash.hex()
}

/// The learning digest: Q-learning and 3-step SARSA trained on Snake from `seed` -- every finished episode, the
/// final Q-tables and a greedy evaluation, hashed. Exercises exploration draws, the update rules and the adapter
/// together, so the determinism fixture covers *learning*, not only arithmetic.
pub fn learning_digest(seed: u64) -> String {
    let mut hash = Fnv::default();
    for (algorithm, n_step) in [("q_learning", 1.0), ("sarsa", 3.0)] {
        let config = TrainerConfig {
            seed,
            seed_pool: (100_000, 1_000_000),
            max_episode_steps: 300,
        };
        let params = Params::new([
            ("epsilon_decay_steps".to_string(), 20_000.0),
            ("n_step".to_string(), n_step),
        ]);
        let factory = factory("snake/features.v1+relative3.v1", 10, 10).unwrap();
        let mut trainer = Trainer::build(factory, algorithm, &params, config).unwrap();
        let stats = trainer.train(30_000);
        for episode in stats.episodes.iter().chain(&trainer.evaluate(&[20_000, 20_001], 500)) {
            hash.f64(episode.total_reward);
            hash.f64(episode.score);
        }
        for byte in trainer.agent().snapshot().bytes() {
            hash.f64(byte as f64);
        }
    }
    hash.hex()
}

/// The DQN digest: a Q-network with every stability piece on (replay, target network, double, dueling, 3-step
/// returns, prioritized replay) trained on Snake's `egocentric.v1` from `seed` -- episodes, a greedy evaluation and the
/// final (folded) network, hashed. Covers backprop, Adam, the sum tree and the ray observer across targets.
pub fn dqn_digest(seed: u64) -> String {
    let mut hash = Fnv::default();
    let config = TrainerConfig {
        seed,
        seed_pool: (100_000, 1_000_000),
        max_episode_steps: 300,
    };
    let params = Params::new(
        [
            ("hidden", 32.0),
            ("learn_start", 500.0),
            ("target_update", 500.0),
            ("epsilon_decay_steps", 4_000.0),
            ("double", 1.0),
            ("dueling", 1.0),
            ("n_step", 3.0),
            ("prioritized", 1.0),
        ]
        .map(|(k, v)| (k.to_string(), v)),
    );
    let factory = factory("snake/egocentric.v1+relative3.v1", 10, 10).unwrap();
    let mut trainer = Trainer::build(factory, "dqn", &params, config).unwrap();
    let stats = trainer.train(6_000);
    for episode in stats.episodes.iter().chain(&trainer.evaluate(&[20_000, 20_001], 300)) {
        hash.f64(episode.total_reward);
        hash.f64(episode.score);
    }
    for byte in trainer.agent().snapshot().bytes() {
        hash.f64(byte as f64);
    }
    hash.hex()
}

/// The policy-gradient digest: PPO on Snake's `egocentric.v1` and on Reach1D (a Gaussian policy), plus REINFORCE with
/// a baseline on Snake, trained from `seed` -- episodes, a greedy evaluation and the final policies, hashed. Covers the
/// softmax and Gaussian heads, GAE, minibatch shuffling and PPO's clipping across targets.
pub fn pg_digest(seed: u64) -> String {
    let mut hash = Fnv::default();
    for (algorithm, env_id) in [
        ("ppo", "snake/egocentric.v1+relative3.v1"),
        ("ppo", "reach1d"),
        ("reinforce", "snake/egocentric.v1+relative3.v1"),
    ] {
        let config = TrainerConfig {
            seed,
            seed_pool: (100_000, 1_000_000),
            max_episode_steps: 300,
        };
        let mut pairs = vec![("hidden".to_string(), 16.0), ("rollout_steps".to_string(), 512.0)];
        if algorithm == "reinforce" {
            pairs.push(("baseline".to_string(), 1.0));
        }
        let params = Params::new(pairs);
        let mut trainer = Trainer::build(factory(env_id, 10, 10).unwrap(), algorithm, &params, config).unwrap();
        let stats = trainer.train(4_000);
        for episode in stats.episodes.iter().chain(&trainer.evaluate(&[20_000, 20_001], 300)) {
            hash.f64(episode.total_reward);
            hash.f64(episode.score);
        }
        for byte in trainer.agent().snapshot().bytes() {
            hash.f64(byte as f64);
        }
    }
    hash.hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factories_parse_ids_and_round_trip() {
        for id in [
            "snake/features.v1+relative3.v1",
            "snake/egocentric.v1+relative3.v1",
            "snake/grid-flat.v1+relative3.v1",
            "reach1d",
        ] {
            assert_eq!(factory(id, 10, 10).unwrap().id(), id);
        }
        assert!(factory("snake/nope.v1+relative3.v1", 10, 10).is_err());
        assert!(factory("chess", 10, 10).is_err());
        assert_eq!(
            factory("snake/grid-flat.v1+relative3.v1", 10, 10)
                .unwrap()
                .make()
                .observation_size(),
            100
        );
    }

    #[test]
    fn same_seed_same_game_and_reach_targets_vary() {
        let snake = factory("snake/features.v1+relative3.v1", 10, 10).unwrap();
        let mut a = snake.make();
        let mut b = snake.make();
        let (ea, eb) = (
            play(a.as_mut(), 42, 50, |_| Action::Discrete(1)),
            play(b.as_mut(), 42, 50, |_| Action::Discrete(1)),
        );
        assert_eq!(ea, eb);
        let mut reach = Reach1DFactory.make();
        let (t1, t2) = (reach.reset(1)[0], reach.reset(2)[0]);
        assert!(t1 != t2 && (-5.0..=5.0).contains(&-t1));
        assert_eq!(
            play(reach.as_mut(), 1, 30, |_| Action::Continuous(0.0)).steps,
            30,
            "never ends on its own"
        );
    }

    #[test]
    fn sparse_reward_keeps_only_food_and_death() {
        let seen = |reward| {
            let mut env = factory_with("snake/features.v1+relative3.v1", 10, 10, reward)
                .unwrap()
                .make();
            env.reset(7);
            let mut rewards = vec![];
            for _ in 0..200 {
                let step = env.step(Action::Discrete(1));
                rewards.push(step.reward);
                if step.done {
                    break;
                }
            }
            rewards
        };
        let (shaped, sparse) = (seen(Reward::Shaped), seen(Reward::Sparse));
        assert_eq!(shaped.len(), sparse.len(), "same game either way");
        for (s, p) in shaped.iter().zip(&sparse) {
            assert_eq!(*p, if s.abs() >= 1.0 { *s } else { 0.0 });
        }
        assert!(shaped.iter().any(|r| r.abs() < 1.0 && *r != 0.0));
        assert!(Reward::parse("dense").is_err());
    }

    #[test]
    fn tabular_learners_get_a_discretizer_only_where_a_table_fits() {
        let rows = |id: &str| factory(id, 10, 10).unwrap().make().discretizer().map(|d| d.states());
        assert_eq!(rows("snake/features.v1+relative3.v1"), Some(2048));
        assert_eq!(rows("snake/egocentric.v1+relative3.v1"), None);
        assert_eq!(rows("snake/grid-flat.v1+relative3.v1"), None);
        assert_eq!(rows("reach1d"), Some(325));
        let config = TrainerConfig {
            seed: 0,
            seed_pool: (0, 10),
            max_episode_steps: 100,
        };
        let refuse = Trainer::build(
            factory("snake/grid-flat.v1+relative3.v1", 10, 10).unwrap(),
            "q_learning",
            &Params::default(),
            config,
        );
        assert!(refuse.err().unwrap().contains("tabular"));
    }

    #[test]
    fn greedy_beats_random_and_digest_is_stable() {
        let snake = factory("snake/features.v1+relative3.v1", 10, 10).unwrap();
        let seeds: Vec<u64> = (10_000..10_020).collect();
        let mean = |name| {
            let episodes = evaluate_baseline(name, snake.as_ref(), &seeds, 1000).unwrap();
            episodes.iter().map(|e| e.score).sum::<f64>() / episodes.len() as f64
        };
        assert!(mean("snake-greedy") > mean("snake-random") + 5.0);
        assert_eq!(rollout_digest(3), rollout_digest(3));
    }
}
