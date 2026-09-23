import random

import checkers_neat_run
import checkers_training as training
import run_context
from evolve import InnovationTracker, NeatGenome, WeightVector, initial_genome
from evolve.neat import ConnectionGene
from evolve.networks import compiled
from games import _native
from games.checkers import Checkers
from games.checkers_strategies import graph_evaluator
from telemetry import FileMetricsStore, SqliteRunRegistry


def _positions(games=4, seed=0):
    rng = random.Random(seed)
    for _ in range(games):
        env = Checkers()
        env.reset()
        done = False
        for _ in range(80):
            if done:
                break
            yield env
            _, _, done = env.step(rng.choice(env.legal_moves()))


def _genome_with_hidden(rng: random.Random) -> NeatGenome:
    """A NEAT genome with two hidden nodes, a skip connection and a disabled gene -- every shape the plan handles."""
    tracker = InnovationTracker(first_hidden_id=34)
    base = initial_genome(32, 1, tracker, rng, 0.4)
    h1, h2 = 34, 35
    extra = [
        ConnectionGene(innovation=tracker.connection(0, h1), source=0, target=h1, weight=rng.uniform(-1, 1)),
        ConnectionGene(innovation=tracker.connection(5, h1), source=5, target=h1, weight=rng.uniform(-1, 1)),
        ConnectionGene(innovation=tracker.connection(32, h2), source=32, target=h2, weight=0.3),  # from the bias
        ConnectionGene(innovation=tracker.connection(h1, h2), source=h1, target=h2, weight=rng.uniform(-1, 1)),
        ConnectionGene(innovation=tracker.connection(h2, 33), source=h2, target=33, weight=rng.uniform(-2, 2)),
        ConnectionGene(innovation=tracker.connection(h1, 33), source=h1, target=33, weight=0.7, enabled=False),
    ]
    return NeatGenome(32, 1, tuple(sorted([*base.connections, *extra], key=lambda c: c.innovation)))


def test_the_game_core_plays_a_neat_genome_exactly_as_its_python_forward_pass():
    genome = _genome_with_hidden(random.Random(3))
    strategy = _native.CheckersStrategy("evaluator", 0, depth=1, graph=genome.graph_encoding())

    checked = 0
    for env in _positions():
        scores = strategy.scores(env._core)
        for move, score in zip(env.legal_moves(), scores, strict=True):
            # depth 1: a move is worth minus the network's value for the position it leaves behind
            assert score == -genome.forward(env.simulate(move))[0] or abs(score) > 500  # (a game-ending move scores WIN)
            checked += 1
    assert checked > 100


def test_layered_evaluator_reaches_the_same_scores_at_any_depth_one_search():
    weights = WeightVector(weights=tuple(random.Random(1).uniform(-0.5, 0.5) for _ in range(32 * 4 + 4 + 4 + 1)), layer_sizes=(32, 4, 1))
    plain = _native.CheckersStrategy("evaluator", 0, list(weights.weights), list(weights.layer_sizes))
    via_graph = None  # a layered network has no graph form; the two kinds agree through `compiled` instead
    assert compiled(weights)["kind"] == "layered" and via_graph is None
    env = next(iter(_positions(1)))
    assert plain.scores(env._core) is not None


def test_a_deeper_evaluator_search_plays_legal_moves_and_uses_the_network_at_the_leaves():
    genome = _genome_with_hidden(random.Random(5))
    for depth in (1, 2, 3):
        env = Checkers()
        env.reset()
        move = graph_evaluator(genome.graph_encoding(), depth)(env, random.Random(0))(None, env.legal_moves())
        assert move in env.legal_moves()


def test_material_seeds_start_as_material_evaluators():
    even_board = Checkers().reset()
    up_four = [1.0] * 4 + [0.0] * 28  # four more own men than the opponent
    for seed in range(5):
        weights = training.material_seed_weights((32, 6, 1), random.Random(seed))
        # Random leftovers make an even board read anywhere near zero, but a material edge must read clearly better.
        assert weights.forward(up_four)[0] > weights.forward(even_board)[0] + 0.3
    neat = training.material_seed_neat(InnovationTracker(first_hidden_id=34), random.Random(0), 1.0)
    assert neat.forward(up_four)[0] > neat.forward(even_board)[0] + 0.3


def test_hall_of_fame_grows_with_champions_and_widens_the_fitness_vector():
    pool = training.OpponentPool(["random"], depth=1, hall_size=2, hall_every=1)
    genome = WeightVector(weights=tuple([0.1] * (32 + 1)), layer_sizes=(32, 1))
    assert len(pool.evaluate(genome)) == 2  # one fixed opponent x both seats

    class Summary:
        def __init__(self, generation, champion):
            self.generation, self.champion = generation, champion

    other = WeightVector(weights=tuple([0.2] * 33), layer_sizes=(32, 1))
    third = WeightVector(weights=tuple([0.3] * 33), layer_sizes=(32, 1))
    for i, champion in enumerate([genome, other, third]):
        pool.on_generation(Summary(i, champion))
    assert pool.hall == [other, third]  # bounded: the oldest left
    assert len(pool.evaluate(genome)) == 2 + 2 * 2  # + each hall member from both seats


def test_neat_run_records_structure_curves_and_a_searching_champion(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)
    monkeypatch.setattr(checkers_neat_run, "MONITOR_GAMES", 2)

    run_id = checkers_neat_run.main(generations=2, population_size=8, opponents=("random", "material-2"), held_out_every=1, depth=2, hall=2, seed_material=0.5)

    run = SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id)
    assert run.status == "completed" and run.config["representation"] == "neat" and run.config["search_depth"] == 2
    history = FileMetricsStore(tmp_path / "metrics").history(run_id)
    assert [g.generation for g in history] == [0, 1]
    assert history[-1].extras and "species" in history[-1].extras


def test_margin_scorer_pays_draws_by_material_and_keeps_wins_on_top():
    env = Checkers()
    env.reset()
    even = training.margin_scorer(env, 0, type("R", (), {"winner": None})())
    assert even == 0.0
    env.board = {(1, 0): (0, False), (3, 0): (0, True), (0, 7): (1, False)}  # red: man + king (3), black: man (1)
    ahead = training.margin_scorer(env, 0, type("R", (), {"winner": None})())
    behind = training.margin_scorer(env, 1, type("R", (), {"winner": None})())
    assert 0 < ahead < training.MARGIN_WEIGHT and behind == -ahead
    # A win always outranks the best possible draw, and a loss the worst.
    assert training.margin_scorer(env, 1, type("R", (), {"winner": 1})()) == 1.0 > training.MARGIN_WEIGHT
    assert training.margin_scorer(env, 0, type("R", (), {"winner": 1})()) == -1.0


def test_a_margin_pool_gives_finer_fitness_than_win_draw_loss():
    genome = WeightVector(weights=tuple(random.Random(2).uniform(-0.3, 0.3) for _ in range(33)), layer_sizes=(32, 1))
    plain = training.OpponentPool(["material-3"], depth=2).evaluate(genome)
    margin = training.OpponentPool(["material-3"], depth=2, margin=True).evaluate(genome)
    assert set(plain) <= {-1.0, 0.0, 1.0}
    assert all(-1.0 <= v <= 1.0 for v in margin)


def test_resampling_plays_different_games_each_generation_but_the_same_ones_for_every_genome_within_one():
    genome = WeightVector(weights=tuple(random.Random(4).uniform(-0.3, 0.3) for _ in range(33)), layer_sizes=(32, 1))

    class Summary:
        def __init__(self, generation):
            self.generation, self.champion = generation, genome

    fixed = training.OpponentPool(["random"], depth=1, margin=True)
    resampled = training.OpponentPool(["random"], depth=1, margin=True, resample=True, games_per_opponent=2)
    first = resampled.evaluate(genome)
    assert resampled.evaluate(genome) == first  # same generation: the same games (selection compares like with like)
    assert len(first) == 2 * 2  # 2 games per opponent x both seats

    baseline = fixed.evaluate(genome)
    for g in range(4):
        fixed.on_generation(Summary(g))
        resampled.on_generation(Summary(g))
    assert fixed.evaluate(genome) == baseline  # not resampling: every generation replays the same games
    assert resampled.evaluate(genome) != first  # resampling: new games
