import type { Snippet } from "~/types/code"

// Learning by Self-Play's snippets. Self-play is Rust (libs/rl/rust/envs/src/selfplay.rs -- natively, and as WASM in
// this page's lab); there is no Python version, so the Python is a translation.
const RUST = "libs/rl/rust/envs/src/selfplay.rs"

export const move: Snippet = {
  pseudo: `// The move the network would play: the one leaving the position worst for the opponent
function best_move(net, game)
    for each legal move m
        value(m) ← −net(the position after m, seen from the opponent's side)
    return the move with the highest value`,
  python: `def best_move(net, game) -> int:
    """The move \`net\` would play: the one leaving the position worst for the opponent."""
    # simulate() encodes the position the move leaves from the *opponent's* side
    values = [-net.forward(game.simulate(move))[0] for move in game.legal_moves()]
    return max(range(len(values)), key=values.__getitem__)   # the first of any ties`,
  rust: {
    source: RUST,
    code: `/// The move \`net\` would play: the one leaving the position worst for the opponent.
fn best_move(net: &Mlp, game: &Checkers) -> usize {
    let moves = game.legal_moves();
    let mut best = (0, f64::NEG_INFINITY);
    for (i, mv) in moves.iter().enumerate() {
        // simulate() encodes the position the move leaves from the *opponent's* side
        let value = -net.forward(&game.simulate(mv).expect("a legal move"))[0];
        if value > best.1 {
            best = (i, value);
        }
    }
    best.0
}`,
  },
}

export const lambda: Snippet = {
  pseudo: `// λ-return targets for one game's positions, walking backwards from the result
function lambda_returns(values, outcome, λ)
    target[last] ← outcome                 // how the game actually ended, for the last mover
    for t from the second-to-last position down to the first
        // the next position belongs to the other player, so its value -- and its target -- flip sign
        target[t] ← −( (1 − λ) × values[t + 1] + λ × target[t + 1] )
    return target`,
  python: `def lambda_returns(values: list[float], outcome: float, lam: float) -> list[float]:
    """λ-return targets for one game's positions, given the network's values of them and the outcome for the
    player who made the last move."""
    n = len(values)
    targets = [0.0] * n
    targets[-1] = outcome                     # the last position: how the game actually ended, for its mover
    for t in reversed(range(n - 1)):
        # the next position belongs to the other player, so its value -- and its target -- flip sign
        targets[t] = -((1 - lam) * values[t + 1] + lam * targets[t + 1])
    return targets`,
  rust: {
    source: RUST,
    code: `/// λ-return targets for one game's positions, given the network's values of them and the outcome for the
/// player who made the last move.
pub fn lambda_returns(values: &[f64], outcome: f64, lambda: f64) -> Vec<f64> {
    let n = values.len();
    let mut targets = vec![0.0; n];
    targets[n - 1] = outcome;                    // the last position: how the game actually ended, for its mover
    for t in (0..n - 1).rev() {
        // the next position belongs to the other player, so its value -- and its target -- flip sign
        targets[t] = -((1.0 - lambda) * values[t + 1] + lambda * targets[t + 1]);
    }
    targets
}`,
  },
}
