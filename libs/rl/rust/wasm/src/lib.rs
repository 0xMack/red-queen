//! The RL core for the browser (docs/design/0010): the same Rust the training jobs run through PyO3, compiled to
//! WebAssembly for Learn's live-training demos: the determinism digests and benchmark kernels (`/dev/rl`, timed
//! from JS with `performance.now()`, since `std::time` doesn't exist here), a `Trainer` a demo drives tick by tick
//! and looks inside (a tabular agent's Q-table and visits), a `DemoGame` to watch the greedy policy play Snake, and a
//! `DemoEnv` for any other environment (Reach1D's continuous control).
//!
//! Built by `libs/rl/build-wasm.py` into `apps/frontend/app/wasm/rl/`.

use redqueen_games::snake::{Label, Observer, Snake};
use redqueen_rl::agent::{Params, Trainer as CoreTrainer, TrainerConfig};
use redqueen_rl::digest;
use redqueen_rl::env::{Action, Env};
use redqueen_rl::nn::{Activation, Shape};
use redqueen_rl_envs as envs;
use wasm_bindgen::prelude::*;

fn shape(layer_sizes: &[u32], activations: &str) -> Result<Shape, JsError> {
    let activations = activations
        .split(',')
        .map(|a| Activation::parse(a.trim()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| JsError::new(&e))?;
    Shape::new(layer_sizes.iter().map(|&n| n as usize).collect(), activations).map_err(|e| JsError::new(&e))
}

/// `training_digest(seed, updates)` of `redqueen_rl::digest`: must equal the native build's (determinism.json).
#[wasm_bindgen(js_name = trainingDigest)]
pub fn training_digest(seed: u32, updates: u32) -> String {
    digest::training_digest(seed as u64, updates)
}

/// `rollout_digest(seed)`: a random agent trained and evaluated on Snake and Reach1D, hashed.
#[wasm_bindgen(js_name = rolloutDigest)]
pub fn rollout_digest(seed: u32) -> String {
    envs::rollout_digest(seed as u64)
}

/// `learning_digest(seed)`: Q-learning and SARSA trained on Snake, hashed.
#[wasm_bindgen(js_name = learningDigest)]
pub fn learning_digest(seed: u32) -> String {
    envs::learning_digest(seed as u64)
}

/// `dqn_digest(seed)`: a DQN with every stability piece trained on Snake's egocentric observer, hashed.
#[wasm_bindgen(js_name = dqnDigest)]
pub fn dqn_digest(seed: u32) -> String {
    envs::dqn_digest(seed as u64)
}

/// `pg_digest(seed)`: PPO (softmax and Gaussian) and REINFORCE with a baseline, trained and hashed.
#[wasm_bindgen(js_name = pgDigest)]
pub fn pg_digest(seed: u32) -> String {
    envs::pg_digest(seed as u64)
}

/// `selfplay_digest(seed)`: TD(λ) Checkers self-play with an opponent pool, trained and hashed.
#[wasm_bindgen(js_name = selfplayDigest)]
pub fn selfplay_digest(seed: u32) -> String {
    envs::selfplay_digest(seed as u64)
}

/// `iterations` training updates (forward + backward + Adam) on a batch. `activations`: comma-separated, one per
/// layer after the input (`"relu,relu,linear"`). Returns a checksum; time the call.
#[wasm_bindgen(js_name = benchUpdates)]
pub fn bench_updates(layer_sizes: &[u32], activations: &str, batch: u32, iterations: u32) -> Result<f64, JsError> {
    Ok(digest::bench_updates(
        &shape(layer_sizes, activations)?,
        batch as usize,
        iterations,
        0,
    ))
}

/// `iterations` single-observation forward passes (choosing an action). Returns a checksum; time the call.
#[wasm_bindgen(js_name = benchForwards)]
pub fn bench_forwards(layer_sizes: &[u32], activations: &str, iterations: u32) -> Result<f64, JsError> {
    Ok(digest::bench_forwards(&shape(layer_sizes, activations)?, iterations, 0))
}

/// `"name=value,name=value"` (empty for the defaults) -> `Params`.
fn parse_params(params: &str) -> Result<Params, JsError> {
    let mut pairs = Vec::new();
    for pair in params.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (name, value) = pair
            .split_once('=')
            .ok_or_else(|| JsError::new(&format!("want name=value, got {pair:?}")))?;
        let value: f64 = value
            .trim()
            .parse()
            .map_err(|_| JsError::new(&format!("not a number: {value:?}")))?;
        pairs.push((name.trim().to_string(), value));
    }
    Ok(Params::new(pairs))
}

/// One agent learning in one environment (`snake/<observer>+relative3.v1` on 10x10, or `reach1d`) -- the engine of
/// Learn's live demos, the same `Trainer` the training jobs drive through PyO3.
#[wasm_bindgen]
pub struct Trainer {
    inner: CoreTrainer,
}

/// What one call to `Trainer::train` did, as numbers JS can read without a serializer.
#[wasm_bindgen]
pub struct Progress {
    /// Episodes that ended during the call, and their mean return and game score (NaN if none ended).
    pub episodes: u32,
    pub mean_return: f64,
    pub mean_score: f64,
    pub total_steps: f64,
    pub total_episodes: f64,
    pub entropy: f64,
    /// Current epsilon (NaN for an agent that doesn't explore that way) and rows ever updated (tabular; NaN otherwise).
    pub epsilon: f64,
    pub states_visited: f64,
    /// A DQN's mean Q(s, a) and loss over this call's updates, and its updates so far (NaN otherwise).
    pub q_mean: f64,
    pub td_loss: f64,
    pub updates: f64,
}

#[wasm_bindgen]
impl Trainer {
    /// `params`: the algorithm's hyperparameters as `name=value` pairs separated by commas (`"alpha=0.1,n_step=3"`,
    /// empty for the defaults). `reward`: `shaped` or `sparse` (Snake).
    #[wasm_bindgen(constructor)]
    pub fn new(algorithm: &str, env_id: &str, seed: u32, params: &str, reward: &str) -> Result<Trainer, JsError> {
        let reward = envs::Reward::parse(reward).map_err(|e| JsError::new(&e))?;
        let factory = envs::factory_with(env_id, 10, 10, reward).map_err(|e| JsError::new(&e))?;
        let config = TrainerConfig {
            seed: seed as u64,
            seed_pool: (100_000, 1_000_000),
            max_episode_steps: 1000,
        };
        let inner =
            CoreTrainer::build(factory, algorithm, &parse_params(params)?, config).map_err(|e| JsError::new(&e))?;
        Ok(Trainer { inner })
    }

    /// Advance by `steps` environment steps.
    pub fn train(&mut self, steps: u32) -> Progress {
        let stats = self.inner.train(steps as u64);
        let n = stats.episodes.len();
        let mean = |f: fn(&redqueen_rl::env::Episode) -> f64| {
            if n == 0 {
                f64::NAN
            } else {
                stats.episodes.iter().map(f).sum::<f64>() / n as f64
            }
        };
        let extra = |name: &str| {
            stats
                .extras
                .iter()
                .find(|(k, _)| k == name)
                .map_or(f64::NAN, |&(_, v)| v)
        };
        Progress {
            episodes: n as u32,
            mean_return: mean(|e| e.total_reward),
            mean_score: mean(|e| e.score),
            total_steps: stats.total_steps as f64,
            total_episodes: stats.total_episodes as f64,
            entropy: stats.entropy,
            epsilon: extra("epsilon"),
            states_visited: extra("states_visited"),
            q_mean: extra("q_mean"),
            td_loss: extra("td_loss"),
            updates: extra("updates"),
        }
    }

    /// Environment steps trained so far.
    #[wasm_bindgen(getter, js_name = totalSteps)]
    pub fn total_steps(&self) -> f64 {
        self.inner.total_steps() as f64
    }

    /// The greedy policy's game score on each of `seeds`, games capped at `max_steps`.
    pub fn evaluate(&mut self, seeds: &[u32], max_steps: u32) -> Vec<f64> {
        let seeds: Vec<u64> = seeds.iter().map(|&s| s as u64).collect();
        self.inner.evaluate(&seeds, max_steps).iter().map(|e| e.score).collect()
    }

    /// The agent's value for each action on `observation` (a table's row, a Q-network's outputs; empty if it has none).
    #[wasm_bindgen(js_name = actionValues)]
    pub fn action_values(&self, observation: &[f64]) -> Vec<f64> {
        self.inner.agent().action_values(observation).unwrap_or_default()
    }

    /// A tabular agent's values, `row * actions + action` (empty for other agents).
    #[wasm_bindgen(js_name = qValues)]
    pub fn q_values(&self) -> Vec<f64> {
        self.inner
            .agent()
            .table()
            .map(|t| t.values.to_vec())
            .unwrap_or_default()
    }

    /// How many updates each row of the table has had (empty for other agents).
    pub fn visits(&self) -> Vec<u32> {
        self.inner
            .agent()
            .table()
            .map(|t| t.visits.to_vec())
            .unwrap_or_default()
    }

    /// The table row `observation` falls in (-1 for a non-tabular agent).
    pub fn row(&self, observation: &[f64]) -> i32 {
        self.inner
            .agent()
            .table()
            .map_or(-1, |t| t.discretizer.index(observation) as i32)
    }

    /// The greedy action on `observation`, as an index (Snake: 0 left, 1 straight, 2 right).
    #[wasm_bindgen(js_name = greedyAction)]
    pub fn greedy_action(&mut self, observation: &[f64]) -> u32 {
        match self.inner.act_greedy(observation) {
            Action::Discrete(i) => i as u32,
            Action::Continuous(_) => 0,
        }
    }

    /// The greedy action on `observation` as a number: an index (discrete) or the continuous value itself.
    #[wasm_bindgen(js_name = greedyValue)]
    pub fn greedy_value(&mut self, observation: &[f64]) -> f64 {
        match self.inner.act_greedy(observation) {
            Action::Discrete(i) => i as f64,
            Action::Continuous(v) => v,
        }
    }

    /// The current policy as its champion JSON (`modelpack.champions`).
    pub fn snapshot(&self) -> String {
        self.inner.agent().snapshot()
    }
}

/// A Snake game for a demo to play the agent's greedy policy on, move by move, and draw.
#[wasm_bindgen]
pub struct DemoGame {
    game: Snake,
    observer: Observer,
}

#[wasm_bindgen]
impl DemoGame {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, observer_id: &str) -> Result<DemoGame, JsError> {
        let observer =
            Observer::from_id(observer_id).ok_or_else(|| JsError::new(&format!("unknown observer {observer_id}")))?;
        Ok(DemoGame {
            game: Snake::new(10, 10, seed as u64, None),
            observer,
        })
    }

    pub fn observation(&self) -> Vec<f64> {
        self.game.encode(self.observer)
    }

    /// Relative action index (0 left, 1 straight, 2 right); returns the reward.
    pub fn step(&mut self, action: u32) -> f64 {
        self.game.step(action as i64 - 1).0
    }

    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        !self.game.alive
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> u32 {
        self.game.score
    }

    /// Cells as `[x, y, label, ...]`, label 0 body / 1 head / 2 food, in render order (like the games module).
    pub fn cells(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for (x, y, label) in self.game.cells() {
            out.extend([
                x,
                y,
                match label {
                    Label::Body => 0,
                    Label::Head => 1,
                    Label::Food => 2,
                },
            ]);
        }
        out
    }
}

/// Any environment the trainers know (`reach1d`, or a Snake interface id), for a demo to step the greedy policy
/// through and draw -- what `DemoGame` is for Snake, without Snake's board.
#[wasm_bindgen]
pub struct DemoEnv {
    env: Box<dyn Env>,
    observation: Vec<f64>,
    done: bool,
}

#[wasm_bindgen]
impl DemoEnv {
    #[wasm_bindgen(constructor)]
    pub fn new(env_id: &str, seed: u32) -> Result<DemoEnv, JsError> {
        let mut env = envs::factory(env_id, 10, 10).map_err(|e| JsError::new(&e))?.make();
        let observation = env.reset(seed as u64);
        Ok(DemoEnv {
            env,
            observation,
            done: false,
        })
    }

    pub fn observation(&self) -> Vec<f64> {
        self.observation.clone()
    }

    /// Take `value` -- an action index for a discrete environment, the action itself for a continuous one. Returns
    /// the reward.
    pub fn step(&mut self, value: f64) -> f64 {
        let action = match self.env.action_space() {
            redqueen_rl::env::ActionSpace::Discrete(_) => Action::Discrete(value as usize),
            redqueen_rl::env::ActionSpace::Continuous { .. } => Action::Continuous(value),
        };
        let step = self.env.step(action);
        self.observation = step.observation;
        self.done = step.done;
        step.reward
    }

    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        self.done
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.env.score()
    }
}

/// Checkers by self-play (docs/design/0010 Phase 4): the same TD(λ) loop the training job runs, for Learn's demo.
#[wasm_bindgen]
pub struct SelfPlayTrainer {
    inner: envs::selfplay::SelfPlay,
}

/// What one call to `SelfPlayTrainer::train` did.
#[wasm_bindgen]
pub struct SelfPlayProgress {
    pub games: f64,
    pub first_wins: f64,
    pub second_wins: f64,
    pub draws: f64,
    pub mean_plies: f64,
    pub loss: f64,
    pub epsilon: f64,
    pub total_games: f64,
}

#[wasm_bindgen]
impl SelfPlayTrainer {
    /// `params` as `Trainer`'s (`"lambda=0.7,hidden=16"`); games are drawn after 40 moves without a capture and cut
    /// at 200 plies, as everywhere else.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, params: &str) -> Result<SelfPlayTrainer, JsError> {
        let inner = envs::selfplay::SelfPlay::new(seed as u64, &parse_params(params)?, 40, 200)
            .map_err(|e| JsError::new(&e))?;
        Ok(SelfPlayTrainer { inner })
    }

    pub fn train(&mut self, games: u32) -> SelfPlayProgress {
        let s = self.inner.train(games as u64);
        SelfPlayProgress {
            games: s.games as f64,
            first_wins: s.first_wins as f64,
            second_wins: s.second_wins as f64,
            draws: s.draws as f64,
            mean_plies: s.mean_plies,
            loss: s.loss,
            epsilon: s.epsilon,
            total_games: self.inner.games_played() as f64,
        }
    }

    /// Points per game against a fixed strategy (`random`, `material-2`, ...), the network searching `depth` plies.
    #[wasm_bindgen(js_name = pointsAgainst)]
    pub fn points_against(&self, opponent: &str, depth: u32, games: u32, seed: u32) -> Result<f64, JsError> {
        self.inner
            .points_against(opponent, depth, games, seed as u64)
            .map_err(|e| JsError::new(&e))
    }

    /// The network as `evolve.WeightVector` JSON (`{"weights", "layer_sizes"}`): what the Checkers stage plays.
    pub fn snapshot(&self) -> String {
        self.inner.snapshot()
    }

    /// [value of the start position, value of the start position a king up], for the side to move.
    pub fn probe(&self) -> Vec<f64> {
        let (a, b) = self.inner.probe();
        vec![a, b]
    }
}

/// `bandit_digest(seed)`: every bandit strategy on every scenario, hashed.
#[wasm_bindgen(js_name = banditDigest)]
pub fn bandit_digest(seed: u32) -> String {
    envs::bandit::bandit_digest(seed as u64)
}

/// A multi-armed bandit game (docs/design/0011), and the strategy playing it -- or `"human"` for a game whose pulls
/// come from the page. The game page races several of these on one seed; the Learn chapter looks inside one.
#[wasm_bindgen]
pub struct BanditRun {
    inner: envs::bandit::BanditRun,
}

#[wasm_bindgen]
impl BanditRun {
    /// `scenario` (`classic`, `two-lamps`, ...), `observer` (`none.v1` or `lamp.v1`), `strategy` (`thompson`, ...,
    /// or `human`), `params` as `name=value,...`.
    #[wasm_bindgen(constructor)]
    pub fn new(scenario: &str, observer: &str, strategy: &str, params: &str, seed: u32) -> Result<BanditRun, JsError> {
        let (scenario, observer) = envs::bandit::parse(scenario, observer).map_err(|e| JsError::new(&e))?;
        let inner = envs::bandit::BanditRun::new(scenario, observer, strategy, &parse_params(params)?, seed as u64)
            .map_err(|e| JsError::new(&e))?;
        Ok(BanditRun { inner })
    }

    /// The strategy's pick for the next pull, without pulling (-1 in a human game).
    pub fn choose(&mut self) -> i32 {
        self.inner.choose().map_or(-1, |a| a as i32)
    }

    /// Pull `arm`; returns its payout. The strategy (if any) learns from it.
    pub fn pull(&mut self, arm: u32) -> Result<f64, JsError> {
        if self.inner.game.done() || arm as usize >= self.inner.game.arms() {
            return Err(JsError::new("the game is over, or no such arm"));
        }
        Ok(self.inner.pull(arm as usize).0)
    }

    /// The strategy's beliefs about situation `row`, flattened: `[values..., spread..., counts..., probabilities...]`
    /// (each `arms` long; probabilities empty for strategies that don't choose by chance). Empty in a human game.
    pub fn beliefs(&self, row: u32) -> Vec<f64> {
        self.inner.beliefs(row as usize).map_or_else(Vec::new, |b| {
            let counts = b.counts.iter().map(|&c| c as f64);
            b.values
                .iter()
                .chain(&b.spread)
                .copied()
                .chain(counts)
                .chain(b.probabilities.iter().copied())
                .collect()
        })
    }

    /// The situation the strategy is in (the lamp, if it sees it).
    #[wasm_bindgen(getter)]
    pub fn row(&self) -> u32 {
        self.inner.row() as u32
    }
    #[wasm_bindgen(getter)]
    pub fn lamp(&self) -> u32 {
        self.inner.game.lamp as u32
    }
    #[wasm_bindgen(getter)]
    pub fn arms(&self) -> u32 {
        self.inner.game.arms() as u32
    }
    #[wasm_bindgen(getter)]
    pub fn budget(&self) -> u32 {
        self.inner.game.budget
    }
    #[wasm_bindgen(getter)]
    pub fn pulls(&self) -> u32 {
        self.inner.game.pulls
    }
    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        self.inner.game.done()
    }
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> f64 {
        self.inner.game.total
    }
    #[wasm_bindgen(getter)]
    pub fn regret(&self) -> f64 {
        self.inner.game.regret()
    }
    #[wasm_bindgen(getter)]
    pub fn efficiency(&self) -> f64 {
        self.inner.game.efficiency()
    }
    /// 0 = no better than pulling at random, 1 = the best arm every pull.
    #[wasm_bindgen(getter)]
    pub fn skill(&self) -> f64 {
        self.inner.game.skill()
    }
    #[wasm_bindgen(getter, js_name = bestPulls)]
    pub fn best_pulls(&self) -> u32 {
        self.inner.game.best_pulls
    }
    #[wasm_bindgen(getter, js_name = bestArm)]
    pub fn best_arm(&self) -> u32 {
        self.inner.game.best_arm() as u32
    }
    pub fn counts(&self) -> Vec<u32> {
        self.inner.game.counts.clone()
    }
    /// Each arm's true mean right now -- for the reveal, never for play.
    pub fn means(&self) -> Vec<f64> {
        self.inner.game.means()
    }

    /// The arms under lamp `context` before any drift, as JSON `[{kind, mean, ...params}]`, and a drifting game's
    /// switch (`{"at": n, "after": [means]}`) -- the end-of-game reveal.
    pub fn reveal(&self) -> String {
        use redqueen_games::bandit::Payout;
        let game = &self.inner.game;
        let arms = |context: usize| {
            let items: Vec<String> = game
                .payouts(context)
                .iter()
                .map(|p| {
                    let extra = match *p {
                        Payout::Bernoulli { p } => format!(r#""p": {p:?}"#),
                        Payout::Gaussian { sd, .. } => format!(r#""sd": {sd:?}"#),
                        Payout::Jackpot { p, prize } => format!(r#""p": {p:?}, "prize": {prize:?}"#),
                        Payout::Fixed { value } => format!(r#""value": {value:?}"#),
                    };
                    format!(r#"{{"kind": "{}", "mean": {:?}, {extra}}}"#, p.kind(), p.mean())
                })
                .collect();
            format!("[{}]", items.join(", "))
        };
        let lamps: Vec<String> = (0..game.scenario.contexts()).map(arms).collect();
        let drift = game.drift().map_or("null".to_string(), |(at, after)| {
            let means: Vec<String> = after.iter().map(|p| format!("{:?}", p.mean())).collect();
            format!(r#"{{"at": {at}, "after": [{}]}}"#, means.join(", "))
        });
        format!(r#"{{"lamps": [{}], "drift": {drift}}}"#, lamps.join(", "))
    }
}

/// A strategy on `count` games from seed `first`, as `[regret, efficiency, skill, best_rate]` per game, flattened -- the
/// scenario comparisons the Learn chapter draws, computed in the reader's browser.
#[wasm_bindgen(js_name = banditEvaluate)]
pub fn bandit_evaluate(
    strategy: &str,
    params: &str,
    scenario: &str,
    observer: &str,
    first: u32,
    count: u32,
) -> Result<Vec<f64>, JsError> {
    let (scenario, observer) = envs::bandit::parse(scenario, observer).map_err(|e| JsError::new(&e))?;
    let seeds: Vec<u64> = (first as u64..first as u64 + count as u64).collect();
    let results = envs::bandit::evaluate(strategy, &parse_params(params)?, scenario, observer, &seeds)
        .map_err(|e| JsError::new(&e))?;
    Ok(results
        .iter()
        .flat_map(|r| [r.regret, r.efficiency, r.skill, r.best_rate])
        .collect())
}
