//! A bandit strategy playing the games crate's `Bandit` (docs/design/0011) -- shared by the PyO3 binding (the
//! evaluation job) and the WASM one (the game page and the Learn chapter), so both play exactly the same games.
//!
//! What the strategy sees is an *interface*, like every game's (doc 0007): `bandit/none.v1+arm.v1` observes nothing
//! (one table row), `bandit/lamp.v1+arm.v1` observes the lamp's colour (a row per colour). A strategy's random draws
//! come from `Rng::new(seed, Explore)`, so a (strategy, game seed) pair is one fixed game on every target.

pub use redqueen_games::bandit::{Bandit, Scenario, SCENARIOS};
use redqueen_rl::agent::Params;
use redqueen_rl::bandit::{build_strategy, BanditStrategy, Beliefs, Hints};
use redqueen_rl::digest::Fnv;
use redqueen_rl::rng::{Rng, Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observer {
    /// Sees nothing: every pull is the same situation.
    None,
    /// Sees the lamp: red (0) or blue (1).
    Lamp,
}

impl Observer {
    pub fn from_id(id: &str) -> Option<Observer> {
        match id {
            "none.v1" => Some(Observer::None),
            "lamp.v1" => Some(Observer::Lamp),
            _ => None,
        }
    }

    pub fn id(&self) -> &'static str {
        match self {
            Observer::None => "none.v1",
            Observer::Lamp => "lamp.v1",
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Observer::None => 1,
            Observer::Lamp => 2,
        }
    }

    pub fn interface(&self) -> String {
        format!("bandit/{}+arm.v1", self.id())
    }
}

/// Parses `scenario` and `observer` ids.
pub fn parse(scenario: &str, observer: &str) -> Result<(Scenario, Observer), String> {
    let s = Scenario::from_id(scenario).ok_or_else(|| format!("unknown bandit scenario {scenario:?}"))?;
    let o = Observer::from_id(observer)
        .ok_or_else(|| format!("unknown bandit observer {observer:?} (none.v1, lamp.v1)"))?;
    Ok((s, o))
}

/// One game, and (unless a person is playing) the strategy playing it.
pub struct BanditRun {
    pub game: Bandit,
    pub observer: Observer,
    strategy: Option<Box<dyn BanditStrategy>>,
    rng: Rng,
}

impl BanditRun {
    /// `strategy` "human" is a game with no strategy: pulls come from outside.
    pub fn new(
        scenario: Scenario,
        observer: Observer,
        strategy: &str,
        params: &Params,
        seed: u64,
    ) -> Result<BanditRun, String> {
        let game = Bandit::new(scenario, seed);
        let strategy = if strategy == "human" {
            None
        } else {
            let hints = Hints {
                arms: game.arms(),
                rows: observer.rows(),
                binary: scenario.binary(),
                scale: game.reward_scale(),
                max_payout: game.max_payout(),
            };
            Some(build_strategy(strategy, &hints, params)?)
        };
        Ok(BanditRun {
            game,
            observer,
            strategy,
            rng: Rng::new(seed, Stream::Explore),
        })
    }

    /// The situation the strategy is in: the lamp, if it can see it.
    pub fn row(&self) -> usize {
        match self.observer {
            Observer::None => 0,
            Observer::Lamp => self.game.lamp,
        }
    }

    /// The strategy's pick for the next pull (not made yet). `None` for a human game.
    pub fn choose(&mut self) -> Option<usize> {
        let row = self.row();
        let rng = &mut self.rng;
        self.strategy.as_mut().map(|s| s.choose(row, rng))
    }

    /// Pull `arm` (the strategy learns from it): (payout, game over).
    pub fn pull(&mut self, arm: usize) -> (f64, bool) {
        let row = self.row();
        let (reward, done) = self.game.pull(arm);
        let next_row = (!done).then(|| self.row());
        if let Some(s) = self.strategy.as_mut() {
            s.update_to(row, arm, reward, next_row);
        }
        (reward, done)
    }

    /// Let the strategy choose and pull: (arm, payout, game over).
    pub fn step(&mut self) -> Option<(usize, f64, bool)> {
        let arm = self.choose()?;
        let (reward, done) = self.pull(arm);
        Some((arm, reward, done))
    }

    pub fn beliefs(&self, row: usize) -> Option<Beliefs> {
        self.strategy.as_ref().map(|s| s.beliefs(row))
    }
}

/// How one game went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameResult {
    pub seed: u64,
    pub total: f64,
    pub regret: f64,
    pub efficiency: f64,
    /// 0 = no better than pulling at random, 1 = the best arm every pull (`Bandit::skill`).
    pub skill: f64,
    /// Share of pulls that went to the best arm of their situation.
    pub best_rate: f64,
}

/// `strategy` playing game `seed` to the end.
pub fn play_game(
    strategy: &str,
    params: &Params,
    scenario: Scenario,
    observer: Observer,
    seed: u64,
) -> Result<GameResult, String> {
    let mut run = BanditRun::new(scenario, observer, strategy, params, seed)?;
    while !run.game.done() {
        run.step().ok_or("a human game can't play itself")?;
    }
    let game = &run.game;
    Ok(GameResult {
        seed,
        total: game.total,
        regret: game.regret(),
        efficiency: game.efficiency(),
        skill: game.skill(),
        best_rate: game.best_pulls as f64 / game.budget as f64,
    })
}

pub fn evaluate(
    strategy: &str,
    params: &Params,
    scenario: Scenario,
    observer: Observer,
    seeds: &[u64],
) -> Result<Vec<GameResult>, String> {
    seeds
        .iter()
        .map(|&seed| play_game(strategy, params, scenario, observer, seed))
        .collect()
}

/// The bandit digest: every strategy on every scenario (lamp-aware on the lamp scenario) for a few games -- totals,
/// regrets and final beliefs, hashed. With the other digests, the determinism fixture: native = Node = browser.
pub fn bandit_digest(seed: u64) -> String {
    let mut hash = Fnv::default();
    for scenario in SCENARIOS {
        let observer = if scenario.contexts() == 2 {
            Observer::Lamp
        } else {
            Observer::None
        };
        for strategy in redqueen_rl::bandit::STRATEGIES {
            for game in 0..3 {
                let mut run =
                    BanditRun::new(scenario, observer, strategy, &Params::default(), seed * 1000 + game).unwrap();
                while !run.game.done() {
                    let (arm, reward, _) = run.step().unwrap();
                    hash.f64(arm as f64);
                    hash.f64(reward);
                }
                hash.f64(run.game.regret());
                for row in 0..observer.rows() {
                    let beliefs = run.beliefs(row).unwrap();
                    for v in beliefs
                        .values
                        .iter()
                        .chain(&beliefs.spread)
                        .chain(&beliefs.probabilities)
                    {
                        hash.f64(*v);
                    }
                }
            }
        }
    }
    hash.hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_regret(strategy: &str, scenario: Scenario, observer: Observer) -> f64 {
        let seeds: Vec<u64> = (0..300).collect();
        let results = evaluate(strategy, &Params::default(), scenario, observer, &seeds).unwrap();
        results.iter().map(|r| r.regret).sum::<f64>() / results.len() as f64
    }

    #[test]
    fn learning_beats_random_and_thompson_beats_greedy_on_the_classic_game() {
        let regret = |s| mean_regret(s, Scenario::Classic, Observer::None);
        let (random, greedy, thompson) = (regret("random"), regret("greedy"), regret("thompson"));
        assert!(
            greedy < random && thompson < greedy * 0.7,
            "{random} {greedy} {thompson}"
        );
    }

    #[test]
    fn seeing_the_lamp_is_what_solves_two_lamps() {
        let blind = mean_regret("thompson", Scenario::TwoLamps, Observer::None);
        let sighted = mean_regret("thompson", Scenario::TwoLamps, Observer::Lamp);
        assert!(sighted < blind * 0.7, "{sighted} vs {blind}");
    }

    #[test]
    fn only_a_strategy_that_looks_ahead_takes_the_detour() {
        let seeds: Vec<u64> = (0..300).collect();
        let skill = |strategy: &str, params: &[(&str, f64)]| {
            let params = Params::new(params.iter().map(|(k, v)| (k.to_string(), *v)));
            let r = evaluate(strategy, &params, Scenario::Detour, Observer::Lamp, &seeds).unwrap();
            r.iter().map(|g| g.skill).sum::<f64>() / r.len() as f64
        };
        let lookahead = [("gamma", 0.9), ("initial_q", 10.0), ("alpha", 0.5), ("epsilon", 0.0)];
        let myopic = [("gamma", 0.0), ("initial_q", 10.0), ("alpha", 0.5), ("epsilon", 0.0)];
        let (ahead, now, thompson) = (
            skill("q_table", &lookahead),
            skill("q_table", &myopic),
            skill("thompson", &[]),
        );
        assert!(
            ahead > 0.4 && ahead > now + 0.25 && ahead > thompson + 0.25,
            "{ahead} {now} {thompson}"
        );
    }

    #[test]
    fn same_seed_same_game_and_humans_pull_by_hand() {
        let a = play_game("ucb1", &Params::default(), Scenario::Jackpot, Observer::None, 5).unwrap();
        let b = play_game("ucb1", &Params::default(), Scenario::Jackpot, Observer::None, 5).unwrap();
        assert_eq!(a, b);
        assert_eq!(bandit_digest(1), bandit_digest(1));
        let mut human = BanditRun::new(Scenario::Classic, Observer::None, "human", &Params::default(), 5).unwrap();
        assert!(human.choose().is_none() && human.beliefs(0).is_none());
        human.pull(3);
        assert_eq!(human.game.pulls, 1);
        assert!(parse("classic", "grid.v1").is_err() && parse("slots", "none.v1").is_err());
        assert_eq!(Observer::Lamp.interface(), "bandit/lamp.v1+arm.v1");
    }
}
