"""Tests for play_match()/MatchFitnessEvaluator's own logic -- uses a tiny local test double
(_NimEnvironment), not games.checkers, so this test suite stays dependency-free the same way
test_neuro.py's _ConstantTargetEnv keeps evolve's tests from needing to import games. Checkers'
own rules are tested in libs/games/tests/test_checkers.py; this file only exercises the
turn-alternation/reward/fitness machinery every MultiAgentEnvironment shares.
"""

from evolve import MatchFitnessEvaluator, WeightVector, play_match


class _NimEnvironment:
    """Classic Nim: players alternate removing 1-3 tokens from a shared pile; whoever takes the
    last token wins. Simple, deterministic, and has a well-known optimal strategy (always leave a
    multiple of 4), which makes it easy to write a test with a known expected winner."""

    def __init__(self, tokens: int = 13):
        self._start_tokens = tokens
        self.reset()

    def reset(self):
        self.tokens = self._start_tokens
        self.player = 0
        self._winner = None
        return self.tokens

    def legal_moves(self):
        return [n for n in (1, 2, 3) if n <= self.tokens]

    def current_player(self):
        return self.player

    def step(self, move):
        self.tokens -= move
        taker = self.player
        self.player = 1 - self.player
        done = self.tokens == 0
        rewards = {}
        if done:
            self._winner = taker
            rewards = {taker: 1.0, 1 - taker: -1.0}
        return self.tokens, rewards, done

    def winner(self):
        return self._winner


def _always_take_one(observation, legal_moves):
    return 1


def _optimal_nim_strategy(observation, legal_moves):
    tokens = observation
    preferred = tokens % 4 or 1  # leave a multiple of 4 for the opponent whenever possible
    return preferred if preferred in legal_moves else legal_moves[0]


def test_play_match_alternates_players_and_reports_the_winner():
    result = play_match(_NimEnvironment(tokens=13), {0: _optimal_nim_strategy, 1: _always_take_one})

    assert result.winner == 0  # the optimal strategy always wins Nim against a fixed opponent
    assert result.moves_played > 0


def test_play_match_stops_at_max_moves_without_a_winner():
    # A pile far bigger than max_moves * 3 can possibly exhaust -- the match must be cut off.
    result = play_match(
        _NimEnvironment(tokens=1000), {0: _always_take_one, 1: _always_take_one}, max_moves=5
    )

    assert result.moves_played == 5
    assert result.winner is None


def test_match_fitness_evaluator_returns_two_values_per_opponent():
    def act(genome, observation, legal_moves):  # genome unused, always takes 1
        return 1

    evaluator = MatchFitnessEvaluator(
        env_factory=lambda: _NimEnvironment(tokens=13),
        opponents=[_always_take_one],
        act=act,
    )

    fitnesses = evaluator.evaluate(genome=None)

    assert len(fitnesses) == 2  # one opponent, played from both seat 0 and seat 1


def test_match_fitness_evaluator_scores_a_dominant_strategy_as_a_clean_win():
    def act(genome, observation, legal_moves):  # genome unused, plays optimally
        return _optimal_nim_strategy(observation, legal_moves)

    evaluator = MatchFitnessEvaluator(
        env_factory=lambda: _NimEnvironment(tokens=13),
        opponents=[_always_take_one],
        act=act,
    )

    fitnesses = evaluator.evaluate(genome=None)

    assert fitnesses == [1.0, 1.0]  # wins regardless of which seat it plays


def test_match_fitness_evaluator_act_receives_the_actual_genome():
    # A minimal sanity check that `act` is called with the real genome, not a placeholder --
    # WeightVector.forward() isn't otherwise exercised by the Nim tests above.
    genome = WeightVector(weights=(0.0,), layer_sizes=(1, 1))
    seen_genomes = []

    def act(g, observation, legal_moves):
        seen_genomes.append(g)
        return legal_moves[0]

    evaluator = MatchFitnessEvaluator(
        env_factory=lambda: _NimEnvironment(tokens=3),
        opponents=[_always_take_one],
        act=act,
    )

    evaluator.evaluate(genome)

    assert all(g is genome for g in seen_genomes)


def test_env_aware_evaluator_binds_each_strategy_to_that_matchs_own_env():
    # Lookahead opponents and position-scoring genomes need the env; the evaluator builds a fresh one
    # per match, so it must hand each strategy the env *its* match is played on.
    envs_made = []
    bound = []

    def env_factory():
        envs_made.append(_NimEnvironment(tokens=5))
        return envs_made[-1]

    def opponent(env):
        bound.append(env)
        return _always_take_one

    def act(genome, env):
        bound.append(env)
        return _optimal_nim_strategy

    evaluator = MatchFitnessEvaluator(env_factory, [opponent], act, env_aware=True)

    assert evaluator.evaluate(genome=None) == [1.0, 1.0]
    assert len(envs_made) == 2
    # per match: the genome's strategy and the opponent are both bound to that match's env
    assert [id(e) for e in bound] == [id(envs_made[0])] * 2 + [id(envs_made[1])] * 2


def test_a_custom_scorer_replaces_win_draw_loss_and_sees_the_finished_env():
    seen = []

    def scorer(env, genome_seat, result):
        seen.append((env.tokens, genome_seat, result.winner))
        return 0.25 if result.winner == genome_seat else -0.25

    def act(genome, observation, legal_moves):
        return _optimal_nim_strategy(observation, legal_moves)

    evaluator = MatchFitnessEvaluator(
        env_factory=lambda: _NimEnvironment(tokens=13), opponents=[_always_take_one], act=act, scorer=scorer
    )

    assert evaluator.evaluate(genome=None) == [0.25, 0.25]
    assert [tokens for tokens, _seat, _winner in seen] == [0, 0]  # the scorer got the finished game
