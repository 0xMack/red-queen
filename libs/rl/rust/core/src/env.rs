//! The one thing the learning core knows about a game (docs/design/0010 Decision 1): `Env`. Adapters for the games
//! crate's Snake and Reach1D live in `redqueen-rl-envs`, so a new game needs an adapter, never a core change.

/// What an agent may do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionSpace {
    /// `n` choices, `0..n`.
    Discrete(usize),
    /// One real number in `[low, high]`.
    Continuous { low: f64, high: f64 },
}

impl ActionSpace {
    /// Output units a network needs to act in this space: one per choice, or a mean (continuous).
    pub fn outputs(&self) -> usize {
        match self {
            ActionSpace::Discrete(n) => *n,
            ActionSpace::Continuous { .. } => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Discrete(usize),
    Continuous(f64),
}

/// The result of one step.
#[derive(Clone, Debug)]
pub struct Step {
    pub observation: Vec<f64>,
    pub reward: f64,
    /// The episode is over (the game ended; a step cap is the caller's business).
    pub done: bool,
}

pub trait Env {
    fn observation_size(&self) -> usize;
    fn action_space(&self) -> ActionSpace;
    /// Start game `seed` from the beginning; returns the first observation. The same seed is the same game.
    fn reset(&mut self, seed: u64) -> Vec<f64>;
    fn step(&mut self, action: Action) -> Step;
    /// The game's own measure of how well the episode went -- what a leaderboard ranks (Snake: apples eaten),
    /// as opposed to the shaped return a learner optimizes.
    fn score(&self) -> f64;
}

/// Makes fresh environments of one configuration: training and evaluation each need their own.
pub trait EnvFactory {
    fn make(&self) -> Box<dyn Env>;
    /// Short id recorded with a run: an interface id (`snake/features.v1+relative3.v1`) or a game name.
    fn id(&self) -> String;
}

/// One finished (or step-capped) episode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Episode {
    pub seed: u64,
    pub total_reward: f64,
    pub steps: u32,
    pub score: f64,
}

/// Plays one episode of game `seed`, `act` choosing every move, for at most `max_steps` steps -- exactly the loop
/// `jobs/evaluate.py`'s `play_episode` runs, so scores through an adapter match the leaderboard's.
pub fn play(env: &mut dyn Env, seed: u64, max_steps: u32, mut act: impl FnMut(&[f64]) -> Action) -> Episode {
    let mut observation = env.reset(seed);
    let (mut total_reward, mut steps) = (0.0, 0);
    while steps < max_steps {
        let step = env.step(act(&observation));
        total_reward += step.reward;
        steps += 1;
        if step.done {
            break;
        }
        observation = step.observation;
    }
    Episode {
        seed,
        total_reward,
        steps,
        score: env.score(),
    }
}
