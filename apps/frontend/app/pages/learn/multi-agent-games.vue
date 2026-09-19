<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
// Round-robin numbers: jobs/checkers_round_robin.py (200 games per pairing, seats alternating).
import { checkersSnapshot } from "~/data/checkersSnapshot"

const protocolCode = `class MultiAgentEnvironment(Protocol):
    def reset(self) -> Observation: ...
    def legal_moves(self) -> list[Move]: ...
    def current_player(self) -> int: ...
    def step(self, move: Move) -> tuple[Observation, dict[int, float], bool]:
        """(observation, {player: reward}, done)"""
    def winner(self) -> int | None: ...`

const strategyCode = `# (observation, legal_moves) -> move. That's the whole interface: a hand-written
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
    return MatchResult(winner=env.winner(), moves_played=moves_played)`

const materialCode = `def material_1(env, rng):
    # simulate() = the position after a move, from the opponent's side (their move next)
    return lambda obs, moves: max((-sum(env.simulate(m)), rng.random(), m) for m in moves)[2]

def material_2(env, rng):
    def pick(obs, moves):
        best = None
        for move in moves:
            child = copy.deepcopy(env)
            child.step(move)
            # assume the opponent replies with whatever is worst for me
            score = min(sum(child.simulate(reply)) for reply in child.legal_moves())
            ...
        return best_move
    return pick`

const fitnessCode = `class MatchFitnessEvaluator:
    def evaluate(self, genome) -> list[float]:
        fitnesses = []
        for opponent in self._opponents:
            for genome_seat in (0, 1):          # both seats: first-move advantage cancels out
                result = play_match(self._env_factory(),
                                    {genome_seat: genome_strategy, 1 - genome_seat: opponent})
                fitnesses.append(0.0 if result.winner is None
                                 else 1.0 if result.winner == genome_seat else -1.0)
        return fitnesses                         # one "test case" per (opponent, seat)`

const results = [
  { a: "random", b: "first-legal", w: 65, d: 66, l: 69 },
  { a: "random", b: "material-1", w: 90, d: 16, l: 94 },
  { a: "first-legal", b: "material-1", w: 57, d: 68, l: 75 },
  { a: "random", b: "material-2", w: 1, d: 4, l: 195 },
  { a: "first-legal", b: "material-2", w: 0, d: 7, l: 193 },
  { a: "material-1", b: "material-2", w: 0, d: 10, l: 190 },
]
</script>

<template>
  <article class="prose-chapter">
    <p>
      Snake has one player. The world mostly doesn't. Games against an opponent change what "fitness"
      even means -- you're not maximizing a score, you're beating whoever is across the board, and how
      good you are depends on who that is. This chapter covers the framework this project uses to pit
      any strategy against any other, exercised on checkers (docs/design/0006).
    </p>

    <h2>From one agent to two</h2>
    <p>
      The single-agent interface is <code>reset()</code> and <code>step(action)</code>. It breaks down
      for a board game in three ways: whose turn it is matters, the set of legal moves changes every
      turn, and there's a reward <em>per player</em>. So two-player games get their own protocol, a
      sibling of the single-agent one:
    </p>
    <CodeBlock lang="python" :code="protocolCode" />
    <p>
      Checkers implements it with real rules: men move diagonally forward, kings both ways, captures are
      <strong>mandatory</strong>, and a piece that can keep jumping must. One design choice keeps the
      protocol simple: a <code>Move</code> is a whole turn -- <code>(from, landing, landing, …)</code> --
      so a triple jump is one <code>step()</code> and players always strictly alternate. Forty moves
      without a capture is a draw.
    </p>
    <figure class="card my-6 grid items-center gap-6 p-5 sm:grid-cols-[minmax(0,260px)_minmax(0,1fr)]">
      <CheckersBoard :state="checkersSnapshot" />
      <figcaption class="text-sm text-fg-muted">
        A real mid-game position produced by <code>games.checkers</code> itself (two random players, 22
        plies), rendered from its <code>render_state()</code> -- the same <code>{width, height, cells}</code>
        shape Snake uses, just with labels like <code>red_king</code> instead of <code>head</code>. What a
        strategy <em>sees</em> is different: the 32 playable squares, from its own side (own man +1, own king
        +2, opponent −1 / −2).
      </figcaption>
    </figure>

    <h2>A strategy is just a function</h2>
    <p>
      There's no strategy class hierarchy. Anything that maps an observation and the legal moves to a
      move is a strategy, and one function plays a match between any two of them:
    </p>
    <CodeBlock lang="python" :code="strategyCode" />

    <h2>What actually makes a player good</h2>
    <p>
      Four hand-written strategies, played against each other: <strong>random</strong>,
      <strong>first-legal</strong> (always the first move on the list), <strong>material-1</strong> (pick the
      move that leaves the best piece balance), and <strong>material-2</strong> (the same, but assuming the
      opponent then replies with its best capture -- a two-move lookahead):
    </p>
    <CodeBlock lang="python" :code="materialCode" />
    <div class="card my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left text-[11px] tracking-wide text-fg-subtle uppercase">
            <th class="px-4 py-2 font-medium">Pairing</th>
            <th class="px-3 py-2 text-right font-medium">Wins</th>
            <th class="px-3 py-2 text-right font-medium">Draws</th>
            <th class="px-3 py-2 text-right font-medium">Losses</th>
            <th class="px-4 py-2 font-medium">Split</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in results" :key="r.a + r.b" class="border-b border-line/50 last:border-0">
            <td class="px-4 py-2"><span class="text-fg">{{ r.a }}</span> <span class="text-fg-subtle">vs</span> {{ r.b }}</td>
            <td class="num px-3 py-2 text-right text-life-300">{{ r.w }}</td>
            <td class="num px-3 py-2 text-right text-fg-subtle">{{ r.d }}</td>
            <td class="num px-3 py-2 text-right text-queen-300">{{ r.l }}</td>
            <td class="px-4 py-2">
              <div class="flex h-2 w-40 overflow-hidden rounded-full bg-raised">
                <div class="bg-life-400" :style="{ width: `${r.w / 2}%` }" />
                <div class="bg-fg-subtle/40" :style="{ width: `${r.d / 2}%` }" />
                <div class="bg-queen-400" :style="{ width: `${r.l / 2}%` }" />
              </div>
            </td>
          </tr>
        </tbody>
      </table>
      <p class="border-t border-line px-4 py-2 text-xs text-fg-subtle">
        200 games per pairing, seats alternating, from the first-named side. Reproduce with
        <code>uv run python jobs/checkers_round_robin.py</code>.
      </p>
    </div>
    <Callout variant="finding" title="One move of lookahead is worth almost nothing -- two is worth everything">
      Material-1 beats random 94 to 90: a coin flip. Because captures are mandatory, whenever a capture
      exists <em>every</em> legal move is a capture, and when none exists no move changes the piece count
      -- so looking one move ahead at material has almost nothing to choose between. Looking one move
      further (does this move leave a piece where the opponent can take it?) wins 190–195 of 200 games
      against every other strategy. In this game, <em>search depth</em> matters far more than a cleverer
      way of scoring positions.
    </Callout>

    <h2>Evolving a player</h2>
    <p>
      To evolve a strategy you need a fitness, and against an opponent the natural one is the match
      result. <code>MatchFitnessEvaluator</code> plays each genome against a pool of reference opponents,
      from <em>both</em> seats, and returns one win/draw/loss per game -- the same per-test-case shape as
      every other evaluator, so lexicase selection works on it unchanged:
    </p>
    <CodeBlock lang="python" :code="fitnessCode" />
    <Callout variant="warning" title="Match outcomes are a noisy signal">
      A small weight-vector population evolved against a randomized opponent with this evaluator
      <em>does</em> improve -- mean fitness about 0.09 → 0.21 over 40 generations. But the first attempt,
      with only 4 matches per genome, showed no clean trend at all: a single match's result says as much
      about the random opponent's luck as about the genome. Repeating that opponent six times in the pool
      (more independent samples, same kind of opponent) made the trend clear. If evolution against match
      results looks stuck, check how noisy each fitness value is before blaming the algorithm.
    </Callout>

    <h2>What's next</h2>
    <p>
      Two pieces remain before checkers joins Snake on the <NuxtLink to="/games">Games</NuxtLink> page.
      A <strong>rating system</strong> (Glicko-2, with fixed reference opponents so the scale doesn't
      drift) to turn a pile of match results into a leaderboard, since there's no single score to rank by.
      And a <strong>board UI</strong>, so a human is just one more strategy -- one whose "function" is you,
      clicking a move (docs/design/0007).
    </p>
  </article>
</template>
