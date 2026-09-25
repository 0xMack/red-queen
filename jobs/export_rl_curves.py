"""Write real runs from a tracked RL experiment as a Learn chapter's recorded curves (docs/design/0010).

Some things a live demo can't show reliably -- a divergence that happens in 1 run in 20 -- and some readers can't
run one at all (no WebAssembly): the chapters show real runs instead, each as its held-out score and (per recording)
the network's mean Q(s, a) or the policy's entropy, against env steps.

  uv run python jobs/export_rl_curves.py              # every recording below
  uv run python jobs/export_rl_curves.py dqn-snake    # -> apps/frontend/app/data/recordings/dqn-snake.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_context import TelemetryStores
from snake_experiment import experiment_runs

RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "apps" / "frontend" / "app" / "data" / "recordings"

# recording -> (experiment, extra curves to keep, [(key, arm, rng seed, label)])
RECORDINGS: dict[str, tuple[str, list[str], list[tuple[str, str, int, str]]]] = {
    # the two DQN runs that diverged, a healthy one, and one run per observer (the DQN chapter)
    "dqn-snake": (
        "rl-dqn-v1",
        ["q_mean"],
        [
            ("naive-diverged", "dqn-naive", 0, "no replay, no target network (seed 0)"),
            ("replay-diverged", "dqn-replay", 3, "replay, no target network (seed 3)"),
            ("healthy", "dqn", 3, "replay + target network (seed 3)"),
            ("egocentric", "dqn", 0, "egocentric.v1 (27 inputs)"),
            ("features", "dqn-features", 0, "features.v1 (11 inputs)"),
            ("grid", "dqn-grid", 0, "grid-flat.v1 (100 inputs)"),
        ],
    ),
    # one run per rung of the policy-gradient ladder, same seed (the policy-gradients chapter)
    "pg-snake": (
        "rl-pg-v1",
        ["entropy"],
        [
            ("reinforce", "pg-reinforce", 0, "REINFORCE"),
            ("baseline", "pg-reinforce-baseline", 0, "REINFORCE + baseline"),
            ("a2c", "pg-a2c", 0, "A2C"),
            ("ppo", "pg-ppo", 0, "PPO"),
            ("ppo-ego2", "pg-ppo-ego2", 0, "PPO, egocentric.v2"),
        ],
    ),
}


def recording(name: str) -> dict:
    experiment, extras, picks = RECORDINGS[name]
    registry, metrics, _ = TelemetryStores.open()
    runs = {(r.config["arm"], r.config["rng_seed"]): r for r in experiment_runs(registry, experiment)}
    out = {}
    for key, arm, seed, label in picks:
        run = runs[(arm, seed)]
        history = metrics.history(run.run_id)
        steps = [int((h.extras or {})["env_steps"]) for h in history]
        entry = {
            "run_id": run.run_id,
            "label": label,
            "score": [
                [s, round(h.held_out_score, 2)]
                for s, h in zip(steps, history, strict=True)
                if h.held_out_score is not None
            ],
        }
        # %.4g keeps a diverged 1.3e10 and a healthy 2.63 alike readable
        if "q_mean" in extras:
            entry["q_mean"] = [
                [s, float(f"{(h.extras or {})['q_mean']:.4g}")] for s, h in zip(steps, history, strict=True)
            ]
        if "entropy" in extras:  # a policy's entropy is the run's `diversity` (docs/design/0010 Decision 3)
            entry["entropy"] = [[s, round(h.diversity, 4)] for s, h in zip(steps, history, strict=True)]
        out[key] = entry
    return {"experiment": experiment, "runs": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export recorded RL curves for the Learn chapters.")
    parser.add_argument("names", nargs="*", default=list(RECORDINGS), help=f"any of {list(RECORDINGS)}")
    args = parser.parse_args()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        out = RECORDINGS_DIR / f"{name}.json"
        out.write_text(json.dumps(recording(name), separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size:,} bytes)")
