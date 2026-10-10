"""Bandit leaderboard evaluation (docs/design/0011): every strategy on every scenario, on held-out games.

The bandit's entrants are *strategies*, not trained models: a bandit learns within one game (its arms are drawn afresh
every game), so there is nothing to train beforehand -- each entrant is an algorithm plus its settings, playing every
game from scratch. Same `EvaluationRecord`s in the same store as every other game, so the game page works unchanged.

Protocol `bandit.skill.v1`: HELD_OUT games (seeds 10,000 onward) of every scenario. A game's **skill** is how much
better than pulling at random the strategy did, in expected payout: 0 = no better than random, 100 = the best arm on
every pull (the core's `Bandit::skill`; expected, so it scores the choices, not the dice). The leaderboard ranks by
skill on `classic`, the default game; `metrics.bandit.scenarios` holds every scenario's (the page's scenario matrix),
and for `two-lamps` both blind (`none.v1`) and seeing the lamp (`lamp.v1`) -- the contextual lesson.

Random and greedy are the baselines (the reference points every other strategy is read against); the rest compete.

Run with: uv run python jobs/evaluate_bandit.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Any

from arena.bandit import GAME, HELD_OUT, PROTOCOL, RANKED, SEED_BASE
from arena.costs import hardware_fingerprint
from games.bandit import SCENARIOS
from jobcore import open_sink
from rl import _native
from telemetry import EvaluationRecord

# (key, strategy, params, label, description, baseline?) -- the params are what the game page runs the entrant with.
ENTRANTS: list[tuple[str, str, dict[str, float], str, str, bool]] = [
    ("random", "random", {}, "Random", "Pulls any machine at random. Learns nothing: the floor.", True),
    (
        "greedy",
        "greedy",
        {},
        "Greedy",
        "Always the machine with the best average so far. Never explores on purpose.",
        True,
    ),
    (
        "epsilon",
        "epsilon_greedy",
        {"epsilon": 0.1},
        "ε-greedy (ε 0.1)",
        "Greedy, but a random machine one pull in ten.",
        False,
    ),
    (
        "epsilon-decay",
        "epsilon_greedy",
        {"epsilon": 0.3, "decay": 100},
        "ε-greedy, decaying",
        "Explores a lot at first (ε 0.3), then less and less, until it only exploits.",
        False,
    ),
    (
        "epsilon-tracking",
        "epsilon_greedy",
        {"epsilon": 0.1, "alpha": 0.2},
        "ε-greedy, constant step",
        "ε-greedy whose estimates weight recent payouts more (step 0.2), so it notices when a machine changes.",
        False,
    ),
    (
        "optimistic",
        "optimistic",
        {},
        "Optimistic start",
        "Assumes every machine pays the maximum until it has tried it.",
        False,
    ),
    ("ucb1", "ucb1", {}, "UCB1", "Adds an uncertainty bonus to each estimate: the textbook constant, c = √2.", False),
    (
        "ucb-tuned",
        "ucb1",
        {"c": 0.5},
        "UCB, tuned (c 0.5)",
        "UCB with a smaller bonus, sized for a 100-pull game.",
        False,
    ),
    (
        "thompson",
        "thompson",
        {},
        "Thompson sampling",
        "Samples a plausible value for every machine from its belief, pulls the best sample.",
        False,
    ),
    (
        "gradient",
        "gradient",
        {"alpha": 0.5},
        "Gradient bandit",
        "Learns preferences, not values: a softmax nudged by payouts against a baseline.",
        False,
    ),
    (
        "q-table",
        "q_table",
        {},
        "Q-table (Q-learning's agent)",
        "The Q-learning chapter's agent itself: gamma 0, ε 0.1, step 0.1.",
        False,
    ),
    (
        "q-lookahead",
        "q_table",
        {"gamma": 0.9, "initial_q": 10.0, "alpha": 0.5, "epsilon": 0.0},
        "Q-learning, looking ahead",
        "The Q-learning update with a future: each value is the payout plus 0.9 of the best value of where it leads.",
        False,
    ),
]


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    skills = [r["skill"] * 100 for r in results]
    sd = statistics.stdev(skills) if len(skills) > 1 else 0.0
    return {
        "skill": statistics.fmean(skills),
        "skill_ci95": 1.96 * sd / len(skills) ** 0.5,
        "efficiency": statistics.fmean(r["efficiency"] for r in results) * 100,
        "regret": statistics.fmean(r["regret"] for r in results),
        "best_rate": statistics.fmean(r["best_rate"] for r in results),
        "scores": [round(s, 1) for s in skills],
    }


def evaluate(strategy: str, params: dict[str, float]) -> tuple[dict[str, dict[str, Any]], float]:
    """Every scenario's summary, and microseconds per pull (timed over everything)."""
    seeds = list(range(SEED_BASE, SEED_BASE + HELD_OUT))
    scenarios: dict[str, dict[str, Any]] = {}
    pulls, start = 0, time.perf_counter()
    for scenario, info in SCENARIOS.items():
        # A sequential game is played seeing the room (a strategy that can't tell the rooms apart has nothing to plan);
        # a contextual one both ways (the lesson); a plain one has nothing to see.
        observers = ["lamp.v1"] if info.sequential else ["none.v1", "lamp.v1"] if info.contexts > 1 else ["none.v1"]
        for observer in observers:
            key = scenario if observer == "none.v1" or info.sequential else f"{scenario}:{observer}"
            scenarios[key] = _summary(_native.bandit_evaluate(strategy, scenario, observer, seeds, params))
            pulls += HELD_OUT * info.budget
    return scenarios, (time.perf_counter() - start) * 1e6 / pulls


def record_for(
    entrant: tuple, hardware: dict[str, Any], run: tuple[str, str, dict | None] | None = None
) -> EvaluationRecord:
    """`run`: (run id, champion ref, the run's measured training cost) for an entrant that came out of a recorded run (an
    evolved strategy). A hand-set strategy trains nothing: it learns within each game."""
    key, strategy, params, label, description, baseline = entrant
    scenarios, us_per_pull = evaluate(strategy, params)
    ranked = scenarios[RANKED]
    scores = ranked["scores"]
    info = SCENARIOS[RANKED]
    return EvaluationRecord(
        game=GAME,
        protocol=PROTOCOL,
        entrant_id=f"run:{run[0]}" if run else f"{'baseline' if baseline else 'strategy'}:{key}",
        entrant_kind="baseline" if baseline else "champion",
        label=label,
        interface="bandit/none.v1+arm.v1",
        run_id=run[0] if run else None,
        champion_ref=run[1] if run else None,
        created_at=time.time(),
        metrics={
            "quality": {
                "n": len(scores),
                "mean": ranked["skill"],
                "ci95": ranked["skill_ci95"],
                "median": statistics.median(scores),
                "min": min(scores),
                "max": max(scores),
                "zero_rate": sum(s <= 0 for s in scores) / len(scores),  # games no better than random
                "mean_steps": float(info.budget),
                "train_mean": None,
                "generalization_gap": None,
                "scores": scores,
            },
            "inference": {
                "encode_us": 0.0,
                "decide_us": us_per_pull,
                "total_us": us_per_pull,
                "parameters": info.arms,  # one estimate per machine (a row of the table)
                "artifact_bytes": 0,
            },
            "training": dict(run[2]) if run and run[2] else {"measured": True, "none": True},
            "model": {
                "description": description,
                "observer_level": 1,
                "note": None,
                "shape": {"algorithm": label},
            },
            "protocol": {
                "held_out_seeds": [SEED_BASE, SEED_BASE + HELD_OUT - 1],
                "episodes": HELD_OUT,
                "max_steps": info.budget,
                "board": {"arms": info.arms},
                "metric": "skill",
            },
            "bandit": {"strategy": strategy, "params": params, "scenarios": scenarios},
        },
        hardware={**hardware, "engine": "rust"},
    )


def evolved_entrants() -> list[tuple[tuple, tuple[str, str, dict | None]]]:
    """Every completed bandit evolution run's final champion (jobs/bandit_evolve_run.py): its evolved settings, as an
    entrant linked to the run that produced it."""
    sink = open_sink()
    out = []
    for run in sink.registry.list_runs():
        config = run.config or {}
        if config.get("game") != GAME or config.get("representation") != "evolved_bandit" or run.status != "completed":
            continue
        history = sink.metrics.history(run.run_id)
        if not history:
            continue
        champion = json.loads(sink.artifacts.get_program(history[-1].champion_ref))
        scenarios = config.get("scenarios", [])
        on = scenarios[0] if len(scenarios) == 1 else f"{len(scenarios)} scenarios"
        settings = ", ".join(f"{k} {v}" for k, v in champion["params"].items())
        entrant = (
            run.run_id,
            champion["strategy"],
            champion["params"],
            f"ε-greedy, evolved on {on}",
            f"ε-greedy with settings evolved over {len(history)} generations on {', '.join(scenarios)}: {settings}.",
            False,
        )
        out.append((entrant, (run.run_id, history[-1].champion_ref, (run.summary or {}).get("cost"))))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    store = open_sink().evaluations
    hardware = hardware_fingerprint()
    records = [record_for(e, hardware) for e in ENTRANTS]
    records += [record_for(e, hardware, run) for e, run in evolved_entrants()]
    for record in records:
        store.put(record)
    for entrant_id in store.prune(GAME, PROTOCOL, {r.entrant_id for r in records}):
        print(f"removed {entrant_id}: no longer an entrant")

    columns = [*SCENARIOS, "two-lamps:lamp.v1"]
    print(f"Skill (0 = random, 100 = best arm every pull) on {HELD_OUT} held-out games per scenario\n")
    print("| strategy | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for record in sorted(records, key=lambda r: -r.metrics["quality"]["mean"]):
        s = record.metrics["bandit"]["scenarios"]
        print(f"| {record.label} | " + " | ".join(f"{s[c]['skill']:.1f}" for c in columns) + " |")


if __name__ == "__main__":
    main()
