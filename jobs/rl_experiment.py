"""Tracked comparisons of reinforcement-learning variants on Snake (docs/design/0010 Decision 4).

Like jobs/snake_experiment.py, but budgeted in *environment steps* rather than generations: every arm trains from
the same rng seeds for the same number of steps, every run is recorded to telemetry tagged `config.experiment` /
`config.arm`, and the report scores each run's *final* champion on the leaderboard's 200 held-out games
(evaluate.HELD_OUT_SEEDS) -- never the best-looking iteration. Every arm names the arm it differs from in exactly one
thing (its `baseline`), so each comparison isolates one idea; the report tests every arm against its baseline with an
exact paired permutation test over the seeds (runs with the same seed are paired).

Phase 1 arms (tabular, snake/features.v1+relative3.v1, 1M steps unless noted):
- `q-learning`       alpha 0.1, gamma 0.95, epsilon 1 -> 0.05 over 100k steps, 1-step: the reference arm
- `sarsa`            on-policy: bootstraps from the action actually taken next
- `q-eps-fast` / `q-eps-slow`   epsilon decays over 20k / 500k steps
- `q-gamma-0.9` / `q-gamma-0.99`
- `q-alpha-0.3`
- `q-optimistic`     initial Q = 2 (above any return a step can see) and almost no random exploration (eps 0.02):
                     optimism alone drives the agent to try what it hasn't
- `q-n3` / `sarsa-n3`   3-step returns
- `q-sparse`         reward = outcomes only (+1 food, -1 death), none of the game's shaping
- `q-long`           5M steps: is 1M enough?

Phase 2 arms (DQN, 1M steps, snake/egocentric.v1+relative3.v1 unless noted; each tested against the arm before it):
- `dqn-naive`        a Q-network trained online: every transition once, in order, no target network
- `dqn-replay`       + experience replay (50k transitions, minibatches of 32)
- `dqn`              + a target network (copied every 2,000 steps): the DQN of Mnih et al. (2015)
- `dqn-double`       + Double DQN: the online net picks the next action, the target net values it
- `dqn-dueling`      + dueling heads (V + A - mean A)
- `dqn-n3`           + 3-step returns
- `dqn-per`          + prioritized replay (alpha 0.6, beta 0.4 -> 1)
- `dqn-features` / `dqn-grid`   `dqn` on features.v1 (11 inputs) / grid-flat.v1 (100): does the observation matter?
- `dqn-long`         `dqn-per` for 5M steps: is 1M enough?
- `dqn-ego2`         `dqn` on egocentric.v2 (egocentric.v1 + reachable space per move, 33 inputs)
- `dqn-ego2-long`    `dqn-long` on egocentric.v2
- `dqn-onehot`       `dqn` on grid-onehot.v1 (304 inputs): is grid-flat.v1's failure its encoding?

Phase 3 arms (policy gradients, 2M steps, egocentric.v1 unless noted; each tested against the arm before it):
- `pg-reinforce`            Monte-Carlo returns, whole episodes per update, no baseline
- `pg-reinforce-baseline`   + a learned state-value baseline
- `pg-a2c`                  + bootstrapping: a critic and GAE(0.95), an update every 128 steps
- `pg-ppo`                  + PPO: 2048-step rollouts reused for 4 epochs of minibatches, ratio clipped to 1 +- 0.2
- `pg-ppo-ego2`             `pg-ppo` on egocentric.v2
- `pg-ppo-ego2-long`        `pg-ppo-ego2` for 10M steps: its curves were still rising at 2M
- `pg-ppo-features16`       PPO on neuroevolution's own net and observation (features.v1, 11->16->3 tanh): gradient vs.
                            evolution on the same architecture (compare with snake_experiment's neuro arms)

  uv run python jobs/rl_experiment.py run    --name NAME --arms q-learning,sarsa --seeds 0-4
  uv run python jobs/rl_experiment.py report --name NAME     # writes run-data/experiments/NAME.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

import rl_run
from evaluate import PROTOCOL, measure_quality
from games import interfaces
from modelpack import QTable, champion_parameters, load_champion
from run_context import RUN_DATA_DIR, TelemetryStores
from snake_experiment import _stats, experiment_runs, parse_seeds
from telemetry import FileArtifactStore, FileMetricsStore, RunInfo

EXPERIMENTS_DIR = RUN_DATA_DIR / "experiments"
INTERFACE = "snake/features.v1+relative3.v1"
EGOCENTRIC = "snake/egocentric.v1+relative3.v1"
STEPS_PER_ITERATION = 50_000
REFERENCE_ARM = "q-learning"


@dataclass(frozen=True)
class Arm:
    algorithm: str
    params: dict[str, float] = field(default_factory=dict)
    steps: int = 1_000_000
    reward: str = "shaped"
    interface: str = INTERFACE
    # the arm this one differs from in one thing, tested against it (None: nothing to compare with)
    baseline: str | None = REFERENCE_ARM


ARMS: dict[str, Arm] = {
    "q-learning": Arm("q_learning", baseline=None),
    "sarsa": Arm("sarsa"),
    "q-eps-fast": Arm("q_learning", {"epsilon_decay_steps": 20_000}),
    "q-eps-slow": Arm("q_learning", {"epsilon_decay_steps": 500_000}),
    "q-gamma-0.9": Arm("q_learning", {"gamma": 0.9}),
    "q-gamma-0.99": Arm("q_learning", {"gamma": 0.99}),
    "q-alpha-0.3": Arm("q_learning", {"alpha": 0.3}),
    "q-optimistic": Arm("q_learning", {"initial_q": 2.0, "epsilon_start": 0.02, "epsilon_end": 0.02}),
    "q-n3": Arm("q_learning", {"n_step": 3}),
    "sarsa-n3": Arm("sarsa", {"n_step": 3}),
    "q-sparse": Arm("q_learning", reward="sparse"),
    "q-long": Arm("q_learning", steps=5_000_000),
}

# The stability ladder: each rung adds one idea to the one before (dqn.rs's defaults are the `dqn` rung).
_LADDER = [
    ("dqn-naive", {"replay_capacity": 0, "target_update": 0}),
    ("dqn-replay", {"target_update": 0}),
    ("dqn", {}),
    ("dqn-double", {"double": 1}),
    ("dqn-dueling", {"double": 1, "dueling": 1}),
    ("dqn-n3", {"double": 1, "dueling": 1, "n_step": 3}),
    ("dqn-per", {"double": 1, "dueling": 1, "n_step": 3, "prioritized": 1}),
]
DQN_ARMS: dict[str, Arm] = {
    name: Arm("dqn", params, interface=EGOCENTRIC, baseline=_LADDER[i - 1][0] if i else None)
    for i, (name, params) in enumerate(_LADDER)
}
DQN_ARMS["dqn-features"] = Arm("dqn", interface=INTERFACE, baseline="dqn")
DQN_ARMS["dqn-grid"] = Arm("dqn", interface="snake/grid-flat.v1+relative3.v1", baseline="dqn")
DQN_ARMS["dqn-long"] = Arm("dqn", _LADDER[-1][1], steps=5_000_000, interface=EGOCENTRIC, baseline="dqn-per")
# the observer follow-up: each changes only the observation of the arm it's compared with
EGOCENTRIC_V2 = "snake/egocentric.v2+relative3.v1"
DQN_ARMS["dqn-ego2"] = Arm("dqn", interface=EGOCENTRIC_V2, baseline="dqn")
DQN_ARMS["dqn-ego2-long"] = Arm("dqn", _LADDER[-1][1], steps=5_000_000, interface=EGOCENTRIC_V2, baseline="dqn-long")
DQN_ARMS["dqn-onehot"] = Arm("dqn", interface="snake/grid-onehot.v1+relative3.v1", baseline="dqn-grid")
ARMS.update(DQN_ARMS)

PG_STEPS = 2_000_000
PG_ARMS: dict[str, Arm] = {
    "pg-reinforce": Arm("reinforce", steps=PG_STEPS, interface=EGOCENTRIC, baseline=None),
    "pg-reinforce-baseline": Arm(
        "reinforce", {"baseline": 1}, steps=PG_STEPS, interface=EGOCENTRIC, baseline="pg-reinforce"
    ),
    "pg-a2c": Arm("a2c", steps=PG_STEPS, interface=EGOCENTRIC, baseline="pg-reinforce-baseline"),
    "pg-ppo": Arm("ppo", steps=PG_STEPS, interface=EGOCENTRIC, baseline="pg-a2c"),
    "pg-ppo-ego2": Arm("ppo", steps=PG_STEPS, interface=EGOCENTRIC_V2, baseline="pg-ppo"),
    "pg-ppo-ego2-long": Arm("ppo", steps=10_000_000, interface=EGOCENTRIC_V2, baseline="pg-ppo-ego2"),
    "pg-ppo-features16": Arm(
        "ppo", {"hidden": 16, "hidden_layers": 1}, steps=PG_STEPS, interface=INTERFACE, baseline=None
    ),
}
ARMS.update(PG_ARMS)


def run_experiment(name: str, arms: list[str], seeds: list[int]) -> None:
    registry = TelemetryStores.open().registry
    for seed in seeds:
        for arm_name in arms:
            done = [
                r
                for r in experiment_runs(registry, name)
                if r.config.get("arm") == arm_name and r.config.get("rng_seed") == seed and r.status == "completed"
            ]
            if done:
                print(f"skip {arm_name} seed {seed}: already completed as {done[0].run_id}", flush=True)
                continue
            arm = ARMS[arm_name]
            iterations = max(1, arm.steps // STEPS_PER_ITERATION)
            print(f"=== {name} · {arm_name} · seed {seed} ({arm.steps:,} steps)", flush=True)
            rl_run.main(
                algorithm=arm.algorithm,
                env_id=arm.interface,
                iterations=iterations,
                steps_per_iteration=STEPS_PER_ITERATION,
                held_out_every=max(1, iterations // 10),
                rng_seed=seed,
                params=dict(arm.params),
                tags={"experiment": name, "arm": arm_name},
                reward=arm.reward,
                snapshot_every=iterations,  # the report only scores the final champion: keep the first and last
            )


# --- Report ------------------------------------------------------------------------------------------------------


def paired_permutation_p(a: list[float], b: list[float]) -> float:
    """Exact two-sided p-value for "arm a and arm b score the same", pairing runs by seed: under the null, each pair's
    difference is equally likely to have either sign, so enumerate all 2^n sign flips."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(sum(diffs))
    flips = list(itertools.product((1, -1), repeat=len(diffs)))
    extreme = sum(
        1 for signs in flips if abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12
    )
    return extreme / len(flips)


def summarize_run(run: RunInfo, metrics: FileMetricsStore, artifacts: FileArtifactStore) -> dict[str, Any]:
    history = metrics.history(run.run_id)
    champion = load_champion(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    interface = interfaces.get(run.config["interface"])
    quality = measure_quality(
        interface,
        lambda _seed: lambda obs: interface.action.decode(champion.forward(obs)),
        training_seeds=[],
    )
    cost = (run.summary or {}).get("cost", {})
    return {
        "run_id": run.run_id,
        "arm": run.config["arm"],
        "rng_seed": run.config["rng_seed"],
        "held_out_mean": quality["mean"],
        "held_out_ci95": quality["ci95"],
        "zero_rate": quality["zero_rate"],
        "env_steps": cost.get("env_steps"),
        "active_s": cost.get("active_s"),
        "parameters": champion_parameters(champion),
        "visited_states": champion.visited_states() if isinstance(champion, QTable) else None,
        # how high the network's value estimates ended up (overestimation shows here), and its loss
        "final_q_mean": (history[-1].extras or {}).get("q_mean"),
        "final_td_loss": (history[-1].extras or {}).get("td_loss"),
        # the monitor curve: held-out score (100 other unseen games) against env steps
        "curve": [
            [int((h.extras or {}).get("env_steps", 0)), h.held_out_score]
            for h in history
            if h.held_out_score is not None
        ],
    }


def build_report(name: str) -> dict[str, Any]:
    registry, metrics, artifacts = TelemetryStores.open()
    runs = [r for r in experiment_runs(registry, name) if r.status == "completed"]
    rows = sorted((summarize_run(r, metrics, artifacts) for r in runs), key=lambda row: (row["arm"], row["rng_seed"]))
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in sorted({row["arm"] for row in rows})}
    by_seed = {arm: {row["rng_seed"]: row["held_out_mean"] for row in mine} for arm, mine in by_arm.items()}
    arms: dict[str, Any] = {}
    for arm, mine in by_arm.items():
        scores = [row["held_out_mean"] for row in mine]
        entry: dict[str, Any] = {
            "n": len(mine),
            "seeds": [row["rng_seed"] for row in mine],
            "held_out_mean": _stats(scores),
            "held_out_per_seed": scores,
            "env_steps": _stats([float(row["env_steps"]) for row in mine if row["env_steps"] is not None]),
            "active_s": _stats([row["active_s"] for row in mine if row["active_s"] is not None]),
        }
        if all(row["visited_states"] is not None for row in mine):
            entry["visited_states"] = _stats([float(row["visited_states"]) for row in mine])
        if all(row["final_q_mean"] is not None for row in mine):
            entry["final_q_mean"] = _stats([row["final_q_mean"] for row in mine])
        baseline = ARMS[arm].baseline if arm in ARMS else None
        reference = by_seed.get(baseline, {}) if baseline else {}
        paired = [(row["held_out_mean"], reference[row["rng_seed"]]) for row in mine if row["rng_seed"] in reference]
        if len(paired) >= 2:
            a, b = zip(*paired, strict=True)
            entry["vs_baseline"] = {
                "arm": baseline,
                "mean_difference": round(statistics.fmean(x - y for x, y in paired), 3),
                "paired_p": round(paired_permutation_p(list(a), list(b)), 4),
                "pairs": len(paired),
            }
        arms[arm] = entry
    return {
        "name": name,
        "protocol": f"{PROTOCOL} (200 held-out games per run, final champion)",
        "setup": {
            "steps_per_iteration": STEPS_PER_ITERATION,
            "arms": {
                arm: {"interface": a.interface, "baseline": a.baseline, "steps": a.steps, "params": a.params}
                for arm, a in ARMS.items()
                if arm in arms
            },
        },
        "arms": arms,
        "runs": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\n## {report['name']} -- {report['protocol']}\n")
    print("| arm | n | held-out (mean ± sd) | min-max | vs baseline (p) | env steps | states visited | time (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, a in report["arms"].items():
        h = a["held_out_mean"]
        vs = a.get("vs_baseline")
        versus = f"{vs['mean_difference']:+.2f} vs {vs['arm']} (p={vs['paired_p']:.3f})" if vs else "--"
        visited = f"{a['visited_states']['mean']:.0f}" if "visited_states" in a else "--"
        print(
            f"| {arm} | {a['n']} | {h['mean']:.2f} ± {h['sd']:.2f} | {h['min']:.1f}-{h['max']:.1f} | {versus} | "
            f"{a['env_steps']['mean']:,.0f} | {visited} | {a['active_s']['mean']:.1f} |"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tracked RL comparisons on Snake (docs/design/0010).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train the (arm, seed) runs not done yet")
    run.add_argument("--name", required=True)
    run.add_argument("--arms", default=",".join(ARMS), help=f"comma list of {sorted(ARMS)}")
    run.add_argument("--seeds", default="0-4")
    report = sub.add_parser("report", help="aggregate finished runs")
    report.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        arms = args.arms.split(",")
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            sys.exit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
        run_experiment(args.name, arms, parse_seeds(args.seeds))
    else:
        result = build_report(args.name)
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        (EXPERIMENTS_DIR / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print_report(result)
        print(f"\nwrote {EXPERIMENTS_DIR / (args.name + '.json')}")


if __name__ == "__main__":
    main()
