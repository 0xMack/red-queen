//! The games crate's environments as the RL core's `Env` (docs/design/0010 Decision 1) -- integration glue, kept out
//! of both the learning core (which never names a game) and the games core (which never knows about learning), and
//! shared by the PyO3 and WASM bindings so neither duplicates it.
//!
//! - **Snake**: any native observer in, `relative3.v1` out (`Discrete(3)`: 0/1/2 = turn left/straight/right, the
//!   argmax order of a 3-output model), the game's own score. Game `seed` is exactly `games.snake.Snake(seed=...)`.
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

pub struct SnakeEnv {
    game: Snake,
    observer: Observer,
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
        let (reward, done) = self.game.step(turn);
        Step {
            observation: self.game.encode(self.observer),
            reward,
            done,
        }
    }

    fn score(&self) -> f64 {
        self.game.score as f64
    }
}

pub struct SnakeFactory {
    pub width: i32,
    pub height: i32,
    pub observer: Observer,
    pub max_steps_without_food: Option<u32>,
}

impl EnvFactory for SnakeFactory {
    fn make(&self) -> Box<dyn Env> {
        Box::new(SnakeEnv {
            game: Snake::new(self.width, self.height, 0, self.max_steps_without_food),
            observer: self.observer,
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
/// board) or `reach1d`.
pub fn factory(env_id: &str, width: i32, height: i32) -> Result<Box<dyn EnvFactory>, String> {
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
