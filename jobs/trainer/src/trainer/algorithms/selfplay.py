"""Checkers by self-play (docs/design/0010 Phase 4, 0015-0017): `td_lambda` and `alphazero`.

`td_lambda`: a position-value network learns by TD(λ) from games against itself (`rl.CheckersSelfPlay`, the Rust loop
in libs/rl/rust/envs/src/selfplay.rs) -- no opponents, no fitness function, no population. It is the same kind of
network the evolved Checkers runs produce (a `WeightVector`, 32 -> H -> 1, tanh), used the same way: scored from the
side to move and searched `depth` plies. So its champions join the versus leaderboard and the Checkers page exactly as
the evolved ones do. `agent: {search_depth: N}` (N > 1) trains by TD-Leaf(λ): self-play searches N plies and each
position learns at its principal variation's leaf.

`alphazero` (`rl.CheckersAlphaZero`, docs/design/0017): a policy-and-value network, PUCT search on every move, the
search's visit counts and the game's outcome as targets. Its champion is still the trunk and value unit as a
`WeightVector` -- an evaluator, recorded and ranked exactly as above -- and the final iteration also stores the whole
two-headed network (`<champion_ref>-net`). Extras add the value and policy losses, how far search moved the root's
value (`search_shift`), the policy's entropy at the start, and, at held-out iterations, the network playing *by
search* against material-2 (`mcts_points`). `init_from` a TD run starts its trunk and value unit from that network.

Each iteration trains `games_per_iteration` games and records, in the fields every run has (Decision 3):
- best/mean/worst fitness: the *return* of self-play -- the first mover's mean outcome (+1 win, 0 draw, -1 loss)
  over the iteration's games (near 0 by symmetry; a drift shows a first-move advantage being learned);
- held-out score: every `held_out_every` iterations, the network searching `depth` plies against the fixed opponents
  the evolved runs are monitored with (trainer.checkers.monitor_score) -- the curve to watch;
- extras: TD loss, mean game length, draws, and the network's value of the start position and of a king up.
"""

from __future__ import annotations

import time

import rl
from arena.checkers import INTERFACE, MONITOR_SEED_BASE
from arena.costs import TrainingCostMeter
from evolve import WeightVector
from jobcore import Sink, recorded_run
from jobcore.algorithms.reinforcement import AlphaZeroParams, SelfPlayParams
from jobcore.specs import TrainSpec
from telemetry import GenerationStats

from trainer.checkers import MAX_MOVES, monitor_score
from trainer.registry import adapter, held_out_every, resolve_init_from

LEARNERS = {"td_lambda": rl.CheckersSelfPlay, "alphazero": rl.CheckersAlphaZero}
MCTS_OPPONENT = "material-2"  # AlphaZero's own player (search + policy) is monitored against it at held-out points
OPPONENTS = ("random", "material-1", "material-2")  # the evolved runs' monitor set
MAX_MOVES_WITHOUT_CAPTURE = 40  # games.checkers.Checkers' default: the draw rule every Checkers game here uses


class _Counters:
    """What TrainingCostMeter reads from a fitness evaluator (`episodes`, `steps`): games and plies here."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0


@adapter("td_lambda", "checkers")
@adapter("alphazero", "checkers")
def train_selfplay(spec: TrainSpec, params: SelfPlayParams, sink: Sink) -> str:
    alphazero = spec.algorithm == "alphazero"
    iterations, every, depth = spec.budget_amount, held_out_every(spec, 10), params.depth
    agent = dict(params.agent)
    trainer = LEARNERS[spec.algorithm](
        spec.seed, params=agent, max_moves_without_capture=MAX_MOVES_WITHOUT_CAPTURE, max_plies=MAX_MOVES
    )
    init_run = resolve_init_from(spec, sink)
    if init_run:
        parent_ref = sink.metrics.history(init_run)[-1].champion_ref
        parent = WeightVector.from_json(sink.artifacts.get_program(parent_ref).decode())
        if alphazero:
            trainer.set_value_network(list(parent.weights), list(parent.layer_sizes))
        else:
            trainer.set_weights(list(parent.weights))
    layer_sizes = WeightVector.from_json(trainer.snapshot()).layer_sizes
    leaf = int(agent.get("search_depth", 1))
    simulations = int(agent.get("simulations", 50))
    if alphazero:
        training = f"AlphaZero-style self-play, {simulations} search simulations per move"
    else:
        training = (
            "self-play"
            + (", opponent pool" if agent.get("pool_every") else "")
            + (", prioritized opponents" if agent.get("pfsp") else "")
            + (f", TD-Leaf searching {leaf} plies" if leaf > 1 else "")
        )
    config = {
        "representation": spec.algorithm,
        "game": "checkers",
        # a position evaluator, searched `search_depth` plies -- the interface the evolved evaluators use
        "interface": INTERFACE,
        "search_depth": depth,
        "paradigm": "reinforcement_learning",
        "layer_sizes": list(layer_sizes),
        "params": agent,
        "iterations": iterations,
        "games_per_iteration": params.games_per_iteration,
        "max_moves": MAX_MOVES,
        "opponents": list(OPPONENTS),  # monitored against, never trained against
        "training": training,
        "held_out_every": every,
        "rng_seed": spec.seed,
        **spec.tags,
    }
    if init_run:
        # The run's own cost is only the continuation; the note says where the rest was spent.
        config["init_run"] = init_run
        config["note"] = f"continued from run {init_run[:8]}"

    counters = _Counters()
    cost = TrainingCostMeter(population_size=1, fitness=counters)

    with recorded_run(config, sink) as run:
        control = run.control_callback(cost)
        for iteration in range(iterations):
            stats = trainer.train(params.games_per_iteration)
            counters.episodes = stats["total_games"]
            counters.steps += round(stats["mean_plies"] * stats["games"])
            first_mover_return = (stats["first_wins"] - stats["second_wins"]) / stats["games"]

            champion_ref = f"{run.run_id}-gen{iteration}"
            snapshot = trainer.snapshot()
            run.artifacts.put_program(champion_ref, snapshot.encode("utf-8"))
            last = iteration == iterations - 1
            if alphazero and last:
                run.artifacts.put_program(f"{champion_ref}-net", trainer.network_json().encode("utf-8"))
            held_out = None
            extras: dict[str, float] = {}
            if iteration % every == 0 or last:
                held_out = monitor_score(
                    WeightVector.from_json(snapshot), OPPONENTS, games=params.monitor_games, depth=depth
                )
                if alphazero:
                    assert isinstance(params, AlphaZeroParams)
                    extras["mcts_points"] = trainer.points_against(
                        MCTS_OPPONENT, simulations, params.mcts_games, MONITOR_SEED_BASE + iteration
                    )
            if alphazero:
                start_value, king_up_value, start_entropy = trainer.probe()
                extras |= {
                    "value_loss": stats["value_loss"],
                    "policy_loss": stats["policy_loss"],
                    # how far the search's value of a position moved from the network's own: what search adds
                    "search_shift": stats["search_shift"],
                    "policy_entropy_start": start_entropy,
                }
            else:
                start_value, king_up_value = trainer.probe()
                extras |= {
                    "epsilon": stats["epsilon"],
                    "pool_games": float(stats["pool_games"]),
                    # how the network fares against its past selves (1 = beats them all): the pool's difficulty
                    "pool_score": stats["pool_score"],
                }
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=iteration,
                    timestamp=time.time(),
                    best_fitness=first_mover_return,
                    mean_fitness=first_mover_return,
                    worst_fitness=first_mover_return,
                    diversity=0.0,  # a greedy value player has no policy entropy to report
                    champion_ref=champion_ref,
                    held_out_score=held_out,
                    extras={
                        "env_steps": float(counters.steps),
                        "games": float(stats["total_games"]),
                        "td_loss": stats["loss"],
                        "mean_game_plies": stats["mean_plies"],
                        "draw_rate": stats["draws"] / stats["games"],
                        "value_start": start_value,
                        "value_king_up": king_up_value,
                        **extras,
                    },
                )
            )
            cost.on_generation(None)
            control(None)
            if held_out is not None:
                print(
                    f"iter {iteration:>3}  games {stats['total_games']:>7,}  loss {stats['loss']:.4f}  "
                    f"plies {stats['mean_plies']:.0f}  held-out {held_out:+.3f}",
                    flush=True,
                )
        run.set_training_summary(cost)
    return run.run_id
