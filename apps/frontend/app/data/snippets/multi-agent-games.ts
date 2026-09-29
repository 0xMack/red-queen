import type { Snippet } from "~/types/code"

// Multi-Agent Games' snippets. The match framework is Python (libs/evolve/src/evolve/match.py); the Checkers
// strategies are Rust (libs/games/rust/core/src/checkers_strategies.rs, shared by training and the browser), with the
// original Python kept as their oracle (libs/games/tests/reference_checkers_strategies.py). Rust for the framework is
// a translation.
const MATCH = "libs/evolve/src/evolve/match.py"

export const strategy: Snippet = {
  pseudo: `// A strategy is any function (observation, legal moves) → move:
// a hand-written heuristic, an evolved network, or a human in a browser all fit.
function play_match(game, strategies, max_moves)
    observation ← game.reset()
    repeat until the game is over or max_moves have been played
        player ← whose turn it is
        move ← strategies[player](observation, the legal moves)
        observation ← game.step(move)
    return the winner (or none, for a draw), and how many moves it took`,
  python: {
    source: MATCH,
    code: `# (observation, legal_moves) -> move. That's the whole interface: a hand-written
# heuristic, an evolved network, or (later) a human in a browser all fit it.
Strategy = Callable[[Observation, list[Move]], Move]

def play_match(env, strategies: dict[int, Strategy], max_moves=200) -> MatchResult:
    observation = env.reset()
    moves_played, done = 0, False
    while not done and moves_played < max_moves:
        player = env.current_player()
        move = strategies[player](observation, env.legal_moves())
        observation, _rewards, done = env.step(move)
        moves_played += 1
    return MatchResult(winner=env.winner(), moves_played=moves_played)`,
  },
  rust: `// (observation, legal_moves) -> move. That's the whole interface: a hand-written
// heuristic, an evolved network, or (later) a human in a browser all fit it.
pub type Strategy<'a> = Box<dyn FnMut(&Observation, &[Move]) -> Move + 'a>;

pub fn play_match(env: &mut impl MultiAgentEnvironment, strategies: &mut [Strategy; 2], max_moves: usize) -> MatchResult {
    let mut observation = env.reset();
    let (mut moves_played, mut done) = (0, false);
    while !done && moves_played < max_moves {
        let player = env.current_player();
        let mv = strategies[player](&observation, &env.legal_moves());
        (observation, _, done) = env.step(mv);
        moves_played += 1;
    }
    MatchResult { winner: env.winner(), moves_played }
}`,
}

export const material: Snippet = {
  pseudo: `// material-1: count the pieces after my move
score(move) ← my pieces − their pieces, in the position the move leaves

// material-2: look one reply further -- assume the opponent answers with whatever is worst for me
score(move) ← a win if the move ends the game in my favour, otherwise
              the minimum, over every reply, of the material after that reply

play the highest-scoring move`,
  python: {
    source: "libs/games/tests/reference_checkers_strategies.py",
    code: `def material_1_scores(env: Checkers) -> list[float]:
    # simulate() encodes the position from the opponent's side, so the mover's material is the negated sum.
    return [-sum(env.simulate(m)) for m in env.legal_moves()]


def material_2_scores(env: Checkers) -> list[float]:
    me = env.current_player()
    scores = []
    for move in env.legal_moves():
        child = copy.deepcopy(env)
        _, _, done = child.step(move)
        if done:
            scores.append(WIN_SCORE if child.winner() == me else 0.0)
        else:
            # After the opponent's reply it's my move again, so simulate() is from my perspective.
            scores.append(min(sum(child.simulate(reply)) for reply in child.legal_moves()))
    return scores`,
  },
  rust: {
    source: "libs/games/rust/core/src/checkers_strategies.rs",
    code: `// simulate() encodes the position from the *opponent's* side (they move next), so a move
// is as good as that position is bad for them.
Kind::Material1 => Some(moves.iter().map(|m| -sum(&game.simulate(m).unwrap())).collect()),
Kind::Material2 => {
    let me = game.current_player;
    Some(
        moves
            .iter()
            .map(|m| {
                let mut child = game.clone();
                if child.step(m).unwrap() {
                    return if child.winner == Some(me) { WIN_SCORE } else { 0.0 };
                }
                // After the opponent's reply it's my move again, so simulate() is from my side.
                child
                    .legal_moves()
                    .iter()
                    .map(|r| sum(&child.simulate(r).unwrap()))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect(),
    )
}`,
  },
}

export const fitness: Snippet = {
  pseudo: `function evaluate(genome)
    results ← empty
    for each reference opponent
        for each seat (first to move, second to move)     // both seats: first-move advantage cancels out
            play a match: genome in that seat, the opponent in the other
            append +1 for a win, −1 for a loss, 0 for a draw
    return results        // one "test case" per (opponent, seat) -- so lexicase works unchanged`,
  python: {
    source: MATCH,
    code: `class MatchFitnessEvaluator:
    def evaluate(self, genome) -> list[float]:
        fitnesses = []
        for opponent in self._opponents:
            for genome_seat in (0, 1):          # both seats: first-move advantage cancels out
                result = play_match(self._env_factory(),
                                    {genome_seat: genome_strategy, 1 - genome_seat: opponent})
                fitnesses.append(0.0 if result.winner is None
                                 else 1.0 if result.winner == genome_seat else -1.0)
        return fitnesses                         # one "test case" per (opponent, seat)`,
  },
  rust: `impl MatchFitnessEvaluator {
    pub fn evaluate(&self, genome: &Genome) -> Vec<f64> {
        let mut fitnesses = Vec::new();
        for opponent in &self.opponents {
            for genome_seat in [0, 1] {         // both seats: first-move advantage cancels out
                let mut strategies = [opponent.strategy(), opponent.strategy()];
                strategies[genome_seat] = self.genome_strategy(genome);
                let result = play_match(&mut (self.env_factory)(), &mut strategies, 200);
                fitnesses.push(match result.winner {
                    None => 0.0,
                    Some(w) if w == genome_seat => 1.0,
                    Some(_) => -1.0,
                });
            }
        }
        fitnesses                               // one "test case" per (opponent, seat)
    }
}`,
}
