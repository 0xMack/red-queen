//! Multi-armed bandits (docs/design/0011): K slot machines with hidden payouts, a fixed budget of pulls, and a set
//! of *scenarios* -- each a structure the arms are drawn into, built to trip a particular strategy up.
//!
//! Everything here is `+ - * /` and comparisons on `f64`, all exactly specified by IEEE 754, so
//! `libs/games/tests/reference_bandit.py` reproduces a game bit for bit (as does the WebAssembly build):
//!
//! - **Streams.** A game draws from separate PCG32 streams, `Pcg32::new(stream_seed(seed, k))`: `k = 0` sets the
//!   arms up, `k = 1` lights the lamps (contextual scenarios), `k = 100 + a` is arm `a`'s payouts. So the n-th pull of
//!   arm `a` pays the same whoever pulls it and whatever else they pulled -- a race between players on one seed is
//!   fair, and "the same seed is the same game" holds for every sequence of pulls.
//! - **Draws.** `uniform = next_u32 / 2^32`; a normal is Irwin-Hall, the sum of 12 uniforms minus 6 (in order);
//!   `shuffle` is Fisher-Yates from the last index down, `j = bounded(i + 1)`.
//! - **Regret** is *expected* (pseudo-)regret: each pull adds the best arm's mean minus the pulled arm's mean, in the
//!   situation (lamp, and before/after a drift) the pull was made in. It measures the choices, not the luck.
//! - **A sequential game** (`Detour`, Level 3 of doc 0011) is the exception: a pull also decides the next room, so
//!   "the best arm of this situation" is no longer the best thing to do. Its yardsticks come from dynamic programming
//!   over the whole budget instead -- the expected total of the best possible play and of random play, both starting in
//!   the red room (`plan`) -- and a pull counts as "best" when it's what the best possible play would do then.

use crate::pcg::Pcg32;

/// SplitMix64's finalizer, as `stream_seed` uses it.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The seed of stream `k` of game `seed`.
pub fn stream_seed(seed: u64, k: u64) -> u64 {
    mix(seed ^ mix(k.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

fn uniform(rng: &mut Pcg32) -> f64 {
    rng.next_u32() as f64 / 4_294_967_296.0
}

fn normal(rng: &mut Pcg32) -> f64 {
    let mut sum = 0.0;
    for _ in 0..12 {
        sum += uniform(rng);
    }
    sum - 6.0
}

fn shuffle<T>(items: &mut [T], rng: &mut Pcg32) {
    for i in (1..items.len()).rev() {
        let j = rng.bounded(i as u32 + 1) as usize;
        items.swap(i, j);
    }
}

/// What one arm pays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Payout {
    /// 1 with probability `p`, else 0 -- a win or a loss.
    Bernoulli { p: f64 },
    /// `mean + sd * z`, z roughly standard normal (Irwin-Hall).
    Gaussian { mean: f64, sd: f64 },
    /// `prize` with probability `p`, else 0: a rare big win.
    Jackpot { p: f64, prize: f64 },
    /// Always `value`.
    Fixed { value: f64 },
}

impl Payout {
    pub fn mean(&self) -> f64 {
        match *self {
            Payout::Bernoulli { p } => p,
            Payout::Gaussian { mean, .. } => mean,
            Payout::Jackpot { p, prize } => p * prize,
            Payout::Fixed { value } => value,
        }
    }

    fn draw(&self, rng: &mut Pcg32) -> f64 {
        match *self {
            Payout::Bernoulli { p } => {
                if uniform(rng) < p {
                    1.0
                } else {
                    0.0
                }
            }
            Payout::Gaussian { mean, sd } => mean + sd * normal(rng),
            Payout::Jackpot { p, prize } => {
                if uniform(rng) < p {
                    prize
                } else {
                    0.0
                }
            }
            Payout::Fixed { value } => value,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Payout::Bernoulli { .. } => "bernoulli",
            Payout::Gaussian { .. } => "gaussian",
            Payout::Jackpot { .. } => "jackpot",
            Payout::Fixed { .. } => "fixed",
        }
    }
}

/// The ways a game's arms can be drawn (docs/design/0011 "Scenarios").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    /// 5 Bernoulli arms, means a jittered ladder over 0.15-0.75, 100 pulls. The default.
    Classic,
    /// 5 Bernoulli arms, the best at 0.55 and the rest in [0.42, 0.50): small gaps need many samples.
    CloseCall,
    /// 5 noisy Gaussian arms (sd 3) with means 4-7: an early lucky payout fools a greedy player.
    LuckyStart,
    /// A jackpot arm (50 with p = 0.02, mean 1.0), a steady arm (always 0.8), three Bernoulli arms (0.3-0.6).
    Jackpot,
    /// Classic arms for 200 pulls, except that somewhere in pulls 50-70 the best arm breaks: from then on it pays
    /// what the worst one does, and the runner-up is best.
    Drifting,
    /// 16 Bernoulli arms (0.05-0.80), 100 pulls: not enough to try everything and still exploit.
    TooManyArms,
    /// A red or blue lamp lights before every pull; under blue each arm pays `0.9 - p` of its red `p` (a contextual
    /// bandit: the best arm under one lamp is the worst under the other).
    TwoLamps,
    /// Two rooms, and a pull decides which one you're in next (Level 3, a tiny MDP). The red room has the classic
    /// ladder except that one machine -- the detour -- pays nothing and lights the gold room; in the gold room every
    /// machine pays 3 with probability 0.4-0.8 (means 1.2-2.4) and leads back to red. Detour-then-gold earns more per
    /// pull than the best red machine, but only a player who values what comes next ever takes the detour.
    Detour,
}

pub const SCENARIOS: [Scenario; 8] = [
    Scenario::Classic,
    Scenario::CloseCall,
    Scenario::LuckyStart,
    Scenario::Jackpot,
    Scenario::Drifting,
    Scenario::TooManyArms,
    Scenario::TwoLamps,
    Scenario::Detour,
];

impl Scenario {
    pub fn id(&self) -> &'static str {
        match self {
            Scenario::Classic => "classic",
            Scenario::CloseCall => "close-call",
            Scenario::LuckyStart => "lucky-start",
            Scenario::Jackpot => "jackpot",
            Scenario::Drifting => "drifting",
            Scenario::TooManyArms => "too-many-arms",
            Scenario::TwoLamps => "two-lamps",
            Scenario::Detour => "detour",
        }
    }

    pub fn from_id(id: &str) -> Option<Scenario> {
        SCENARIOS.into_iter().find(|s| s.id() == id)
    }

    pub fn arms(&self) -> usize {
        if *self == Scenario::TooManyArms {
            16
        } else {
            5
        }
    }

    pub fn budget(&self) -> u32 {
        if *self == Scenario::Drifting {
            200
        } else {
            100
        }
    }

    /// Lamps (situations) the arms depend on: 2 for `TwoLamps` and `Detour` (its rooms), 1 otherwise.
    pub fn contexts(&self) -> usize {
        if matches!(self, Scenario::TwoLamps | Scenario::Detour) {
            2
        } else {
            1
        }
    }

    /// Every payout is 0 or 1 (a win or a loss) -- what a Beta-Bernoulli belief needs.
    pub fn binary(&self) -> bool {
        !matches!(self, Scenario::LuckyStart | Scenario::Jackpot | Scenario::Detour)
    }

    /// A pull decides the next situation (the lamp is a room you walk into, not a coin toss).
    pub fn sequential(&self) -> bool {
        *self == Scenario::Detour
    }
}

/// The classic ladder, jittered by +-0.05 (one draw per rung, in order), then shuffled.
fn ladder(rng: &mut Pcg32) -> Vec<Payout> {
    let mut arms: Vec<Payout> = [0.15, 0.30, 0.45, 0.60, 0.75]
        .iter()
        .map(|base| Payout::Bernoulli {
            p: base - 0.05 + 0.1 * uniform(rng),
        })
        .collect();
    shuffle(&mut arms, rng);
    arms
}

fn best_index(arms: &[Payout]) -> usize {
    let mut best = 0;
    for (i, arm) in arms.iter().enumerate() {
        if arm.mean() > arms[best].mean() {
            best = i;
        }
    }
    best
}

/// The yardsticks of a sequential game, by backward induction over the budget from the red room: the expected total of
/// the best possible play and of uniformly random play, and the best arm at every (pull, room).
#[derive(Clone, Debug)]
struct Plan {
    optimal: f64,
    random: f64,
    /// `best[t][room]`: the arm the best possible play pulls at pull `t` in `room` (ties to the lowest index).
    best: Vec<[usize; 2]>,
}

/// Where arm `arm` in `room` leads in the detour game.
fn detour_next(room: usize, arm: usize, detour: usize) -> usize {
    if room == 0 && arm == detour {
        1
    } else {
        0
    }
}

fn plan(arms: &[Vec<Payout>], detour: usize, budget: u32) -> Plan {
    let k = arms[0].len();
    let (mut v, mut r) = ([0.0f64; 2], [0.0f64; 2]);
    let mut best = vec![[0usize; 2]; budget as usize];
    for t in (0..budget as usize).rev() {
        let (mut nv, mut nr) = ([0.0f64; 2], [0.0f64; 2]);
        for room in 0..2 {
            let mut total = 0.0;
            for (arm, payout) in arms[room].iter().enumerate() {
                let next = detour_next(room, arm, detour);
                let q = payout.mean() + v[next];
                if arm == 0 || q > nv[room] {
                    nv[room] = q;
                    best[t][room] = arm;
                }
                total += payout.mean() + r[next];
            }
            nr[room] = total / k as f64;
        }
        (v, r) = (nv, nr);
    }
    Plan {
        optimal: v[0],
        random: r[0],
        best,
    }
}

fn worst_index(arms: &[Payout]) -> usize {
    let mut worst = 0;
    for (i, arm) in arms.iter().enumerate() {
        if arm.mean() < arms[worst].mean() {
            worst = i;
        }
    }
    worst
}

#[derive(Clone, Debug)]
pub struct Bandit {
    pub scenario: Scenario,
    pub seed: u64,
    /// `arms[context][arm]` before any drift.
    arms: Vec<Vec<Payout>>,
    /// A drifting game's arms after the switch, and the pull it happens at (0-based: pull `at` is the first after).
    drift: Option<(u32, Vec<Payout>)>,
    /// A sequential game's detour machine (in the red room), and its yardsticks.
    detour: Option<usize>,
    plan: Option<Plan>,
    arm_rngs: Vec<Pcg32>,
    lamp_rng: Pcg32,
    pub budget: u32,
    pub pulls: u32,
    pub lamp: usize,
    /// Realized payout so far.
    pub total: f64,
    /// Sum of the pulled arms' means, of the best arm's, and of the average arm's (what pulling at random expects),
    /// over the pulls made.
    pub expected: f64,
    pub best_expected: f64,
    pub random_expected: f64,
    /// Pulls of each arm.
    pub counts: Vec<u32>,
    /// Pulls that went to the best arm of their situation.
    pub best_pulls: u32,
}

impl Bandit {
    pub fn new(scenario: Scenario, seed: u64) -> Bandit {
        let mut setup = Pcg32::new(stream_seed(seed, 0));
        let k = scenario.arms();
        let mut drift = None;
        let mut detour = None;
        let arms: Vec<Vec<Payout>> = match scenario {
            Scenario::Classic => vec![ladder(&mut setup)],
            Scenario::CloseCall => {
                let mut arms = vec![Payout::Bernoulli { p: 0.55 }];
                for _ in 0..4 {
                    arms.push(Payout::Bernoulli {
                        p: 0.42 + 0.08 * uniform(&mut setup),
                    });
                }
                shuffle(&mut arms, &mut setup);
                vec![arms]
            }
            Scenario::LuckyStart => {
                let mut arms: Vec<Payout> = [4.0, 4.75, 5.5, 6.25, 7.0]
                    .iter()
                    .map(|&mean| Payout::Gaussian { mean, sd: 3.0 })
                    .collect();
                shuffle(&mut arms, &mut setup);
                vec![arms]
            }
            Scenario::Jackpot => {
                let mut arms = vec![Payout::Jackpot { p: 0.02, prize: 50.0 }, Payout::Fixed { value: 0.8 }];
                for _ in 0..3 {
                    arms.push(Payout::Bernoulli {
                        p: 0.3 + 0.3 * uniform(&mut setup),
                    });
                }
                shuffle(&mut arms, &mut setup);
                vec![arms]
            }
            Scenario::Drifting => {
                let before = ladder(&mut setup);
                let at = 50 + setup.bounded(21);
                let mut after = before.clone();
                after[best_index(&before)] = before[worst_index(&before)];
                drift = Some((at, after));
                vec![before]
            }
            Scenario::TooManyArms => vec![(0..k)
                .map(|_| Payout::Bernoulli {
                    p: 0.05 + 0.75 * uniform(&mut setup),
                })
                .collect()],
            Scenario::TwoLamps => {
                // Blue mirrors red (p -> 0.9 - p): the best machine under one lamp is the worst under the other, and
                // every machine averages 0.45 over both -- a player who can't see the lamp has nothing to find.
                let red = ladder(&mut setup);
                let blue = red
                    .iter()
                    .map(|arm| Payout::Bernoulli { p: 0.9 - arm.mean() })
                    .collect();
                vec![red, blue]
            }
            Scenario::Detour => {
                let mut red = ladder(&mut setup);
                let door = setup.bounded(k as u32) as usize;
                red[door] = Payout::Fixed { value: 0.0 };
                let mut gold: Vec<Payout> = [0.5, 0.6, 0.7, 0.85, 1.0]
                    .iter()
                    .map(|f| Payout::Jackpot { p: 0.8 * f, prize: 3.0 })
                    .collect();
                shuffle(&mut gold, &mut setup);
                detour = Some(door);
                vec![red, gold]
            }
        };
        let plan = detour.map(|door| plan(&arms, door, scenario.budget()));
        let mut game = Bandit {
            scenario,
            seed,
            arms,
            drift,
            detour,
            plan,
            arm_rngs: Vec::new(),
            lamp_rng: Pcg32::new(0),
            budget: scenario.budget(),
            pulls: 0,
            lamp: 0,
            total: 0.0,
            expected: 0.0,
            best_expected: 0.0,
            random_expected: 0.0,
            counts: vec![0; k],
            best_pulls: 0,
        };
        game.reset();
        game
    }

    /// Back to the first pull of the same game: same arms, same payouts in the same order.
    pub fn reset(&mut self) {
        let k = self.scenario.arms();
        self.arm_rngs = (0..k)
            .map(|a| Pcg32::new(stream_seed(self.seed, 100 + a as u64)))
            .collect();
        self.lamp_rng = Pcg32::new(stream_seed(self.seed, 1));
        self.pulls = 0;
        self.total = 0.0;
        self.expected = 0.0;
        self.best_expected = 0.0;
        self.random_expected = 0.0;
        self.counts = vec![0; k];
        self.best_pulls = 0;
        self.lamp = self.next_lamp();
    }

    fn next_lamp(&mut self) -> usize {
        // One draw per pull, and only in a contextual scenario (a plain game's lamp stream is never touched). A
        // sequential game starts in the red room and moves by its own rule (`pull`).
        if self.scenario.contexts() == 1 || self.scenario.sequential() || uniform(&mut self.lamp_rng) < 0.5 {
            0
        } else {
            1
        }
    }

    pub fn arms(&self) -> usize {
        self.scenario.arms()
    }

    pub fn done(&self) -> bool {
        self.pulls >= self.budget
    }

    /// The arms as they pay right now: under the current lamp, after the drift if it has happened.
    pub fn current(&self) -> &[Payout] {
        match &self.drift {
            Some((at, after)) if self.pulls >= *at => after,
            _ => &self.arms[self.lamp],
        }
    }

    /// Each arm's true mean right now -- hidden from players until the game ends.
    pub fn means(&self) -> Vec<f64> {
        self.current().iter().map(Payout::mean).collect()
    }

    /// The arms under lamp `context` before any drift (the reveal shows them all).
    pub fn payouts(&self, context: usize) -> &[Payout] {
        &self.arms[context]
    }

    /// The drift, if this game has one: the pull it happens at, and the arms after it.
    pub fn drift(&self) -> Option<(u32, &[Payout])> {
        self.drift.as_ref().map(|(at, after)| (*at, after.as_slice()))
    }

    /// The best arm now: of this situation -- or, in a sequential game, what the best possible play pulls now.
    pub fn best_arm(&self) -> usize {
        match &self.plan {
            Some(plan) => plan.best[(self.pulls as usize).min(plan.best.len() - 1)][self.lamp],
            None => best_index(self.current()),
        }
    }

    /// A sequential game's detour machine, if this is one.
    pub fn detour(&self) -> Option<usize> {
        self.detour
    }

    /// Pull arm `arm`: its payout, and whether that was the last pull. Pulling after the game is over is a bug.
    pub fn pull(&mut self, arm: usize) -> (f64, bool) {
        assert!(!self.done(), "the game is over");
        assert!(arm < self.arms(), "no arm {arm}");
        let best = self.best_arm();
        let current = self.current();
        let payout = current[arm];
        let (mean, best_mean) = (payout.mean(), current[best].mean());
        let mut average = 0.0;
        for arm in current {
            average += arm.mean();
        }
        let average = average / current.len() as f64;
        let reward = payout.draw(&mut self.arm_rngs[arm]);
        self.counts[arm] += 1;
        self.total += reward;
        self.expected += mean;
        self.best_expected += best_mean;
        self.random_expected += average;
        if arm == best {
            self.best_pulls += 1;
        }
        self.pulls += 1;
        self.lamp = match self.detour {
            Some(door) => detour_next(self.lamp, arm, door),
            None => self.next_lamp(),
        };
        (reward, self.done())
    }

    /// The yardsticks so far: (best possible, random) expected payout over the pulls made. A sequential game's are its
    /// plan's totals, pro-rated by the share of the budget used (exact once the game is over).
    fn yardsticks(&self) -> (f64, f64) {
        match &self.plan {
            Some(plan) => {
                let used = self.pulls as f64 / self.budget as f64;
                (plan.optimal * used, plan.random * used)
            }
            None => (self.best_expected, self.random_expected),
        }
    }

    /// Expected payout given up so far against the best possible play.
    pub fn regret(&self) -> f64 {
        self.yardsticks().0 - self.expected
    }

    /// Expected payout as a share of the best possible: 1.0 = played perfectly.
    pub fn efficiency(&self) -> f64 {
        let best = self.yardsticks().0;
        if best > 0.0 {
            self.expected / best
        } else {
            1.0
        }
    }

    /// How much better than pulling at random: 0 = no better, 1 = the best arm every pull (negative is worse than
    /// random). The leaderboard's score -- comparable across scenarios, where raw payouts and efficiency are not.
    pub fn skill(&self) -> f64 {
        let (best, random) = self.yardsticks();
        let room = best - random;
        if room > 0.0 {
            (self.expected - random) / room
        } else {
            1.0
        }
    }

    /// The largest payout an arm of this game can make, for a strategy that starts optimistic.
    pub fn max_payout(&self) -> f64 {
        match self.scenario {
            Scenario::LuckyStart => 13.0, // the best mean plus two sd
            Scenario::Jackpot => 50.0,
            Scenario::Detour => 3.0,
            _ => 1.0,
        }
    }

    /// The payouts' typical spread, for strategies whose exploration bonus needs a scale (UCB, Gaussian Thompson).
    pub fn reward_scale(&self) -> f64 {
        match self.scenario {
            Scenario::LuckyStart => 3.0,
            Scenario::Jackpot => 7.0, // the jackpot arm's sd: 50 * sqrt(0.02 * 0.98)
            Scenario::Detour => 1.2,  // a gold machine's sd: 3 * sqrt(p (1 - p)), p 0.4-0.8
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play_out(game: &mut Bandit, mut choose: impl FnMut(u32) -> usize) -> Vec<f64> {
        let mut rewards = vec![];
        while !game.done() {
            rewards.push(game.pull(choose(game.pulls)).0);
        }
        rewards
    }

    #[test]
    fn same_seed_same_game_and_payouts_belong_to_the_arm() {
        for scenario in SCENARIOS {
            let mut a = Bandit::new(scenario, 7);
            let mut b = Bandit::new(scenario, 7);
            assert_eq!(a.means(), b.means());
            let k = scenario.arms();
            let ra = play_out(&mut a, |t| t as usize % k);
            // b pulls in a different order: arm 0's n-th payout is still the same
            let rb = play_out(&mut b, |t| if t % 2 == 0 { 0 } else { t as usize % k });
            let arm0 = |rewards: &[f64], order: &dyn Fn(usize) -> usize| {
                (0..rewards.len())
                    .filter(|&t| order(t) == 0)
                    .map(|t| rewards[t])
                    .collect::<Vec<_>>()
            };
            let (xa, xb) = (
                arm0(&ra, &|t| t % k),
                arm0(&rb, &|t| if t % 2 == 0 { 0 } else { t % k }),
            );
            let n = xa.len().min(xb.len());
            assert!(n >= 5, "{scenario:?}: {n}");
            if scenario.contexts() == 1 && scenario != Scenario::Drifting {
                assert_eq!(xa[..n], xb[..n], "{scenario:?}");
            }
            a.reset();
            assert_eq!(play_out(&mut a, |t| t as usize % k), ra, "reset replays the game");
        }
    }

    #[test]
    fn every_scenario_keeps_its_structure_across_seeds() {
        for seed in 0..300 {
            let classic = Bandit::new(Scenario::Classic, seed).means();
            let mut sorted = classic.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(sorted[4] - sorted[3] > 0.04, "classic's best arm stands out");

            let close = Bandit::new(Scenario::CloseCall, seed).means();
            assert_eq!(close.iter().filter(|&&m| m == 0.55).count(), 1);
            assert!(close.iter().all(|&m| m == 0.55 || (0.42..0.5).contains(&m)));

            let jackpot = Bandit::new(Scenario::Jackpot, seed);
            assert_eq!(jackpot.current()[jackpot.best_arm()].kind(), "jackpot");

            let mut drifting = Bandit::new(Scenario::Drifting, seed);
            let (at, _) = drifting.drift().unwrap();
            assert!((50..=70).contains(&at));
            let before = drifting.best_arm();
            play_out(&mut drifting, |_| 0);
            assert_ne!(drifting.best_arm(), before, "the best arm is broken after the drift");

            let lamps = Bandit::new(Scenario::TwoLamps, seed);
            assert_ne!(best_index(lamps.payouts(0)), best_index(lamps.payouts(1)));

            assert_eq!(Bandit::new(Scenario::TooManyArms, seed).arms(), 16);
        }
    }

    #[test]
    fn the_detour_pays_only_for_a_player_who_looks_ahead() {
        for seed in 0..200 {
            let game = Bandit::new(Scenario::Detour, seed);
            let door = game.detour().unwrap();
            assert_eq!(game.payouts(0)[door], Payout::Fixed { value: 0.0 });
            assert!(game.payouts(1).iter().all(|p| p.kind() == "jackpot" && p.mean() >= 1.2));
            // The best possible play takes the detour; staying on the best red machine is worse.
            let plan = game.plan.as_ref().unwrap();
            assert_eq!(plan.best[0][0], door);
            let mut stay = game.clone();
            let red_best = best_index(stay.payouts(0));
            play_out(&mut stay, |_| red_best);
            assert!(stay.skill() < 0.9 && stay.skill() > 0.0, "{}", stay.skill());
            // Following the plan is skill 1, and walks between the rooms.
            let mut perfect = game.clone();
            let mut rooms = vec![];
            while !perfect.done() {
                rooms.push(perfect.lamp);
                let arm = perfect.best_arm();
                perfect.pull(arm);
            }
            assert!((perfect.skill() - 1.0).abs() < 1e-9, "{}", perfect.skill());
            assert!(rooms.contains(&1) && rooms[0] == 0);
            assert_eq!(perfect.best_pulls, 100);
        }
    }

    #[test]
    fn regret_counts_choices_not_luck() {
        let mut game = Bandit::new(Scenario::Classic, 3);
        let best = game.best_arm();
        play_out(&mut game, |_| best);
        assert_eq!(game.regret(), 0.0);
        assert_eq!(game.efficiency(), 1.0);
        assert_eq!(game.skill(), 1.0);
        assert_eq!(game.best_pulls, 100);
        let mut worse = Bandit::new(Scenario::Classic, 3);
        let other = (best + 1) % 5;
        play_out(&mut worse, |_| other);
        let gap = worse.means()[best] - worse.means()[other];
        assert!((worse.regret() - 100.0 * gap).abs() < 1e-9);
    }

    #[test]
    fn lamps_light_both_colours_and_payout_draws_look_right() {
        let mut game = Bandit::new(Scenario::TwoLamps, 11);
        let mut blue = 0;
        while !game.done() {
            blue += game.lamp;
            game.pull(0);
        }
        assert!((30..70).contains(&blue));
        let mut rng = Pcg32::new(5);
        let (mut sum, mut sq, n) = (0.0, 0.0, 20_000);
        for _ in 0..n {
            let z = normal(&mut rng);
            sum += z;
            sq += z * z;
        }
        assert!((sum / n as f64).abs() < 0.03 && (sq / n as f64 - 1.0).abs() < 0.05);
        assert_eq!(Scenario::from_id("two-lamps"), Some(Scenario::TwoLamps));
        assert!(SCENARIOS.iter().all(|s| Scenario::from_id(s.id()) == Some(*s)));
    }
}
