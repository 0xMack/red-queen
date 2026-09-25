"""Write real DQN runs from the `rl-dqn-v1` experiment as the DQN chapter's recorded curves (docs/design/0010 Phase 2b).

Divergence is rare (about 1 run in 20 without a target network), so a reader's live demo almost never shows one --
the chapter shows the real ones instead, next to a healthy run: each run's held-out score and its network's mean
Q(s, a) against env steps. The observer comparison and the no-WebAssembly fallback come from the same runs.

  uv run python jobs/export_dqn_recording.py      # -> apps/frontend/app/data/recordings/dqn-snake.json
"""

from __future__ import annotations

import json
from pathlib import Path

from run_context import TelemetryStores
from snake_experiment import experiment_runs

OUT = Path(__file__).resolve().parents[1] / "apps" / "frontend" / "app" / "data" / "recordings" / "dqn-snake.json"
EXPERIMENT = "rl-dqn-v1"
# (key, arm, rng seed, label): the two runs that diverged, a healthy one, and one run per observer
PICKS = [
    ("naive-diverged", "dqn-naive", 0, "no replay, no target network (seed 0)"),
    ("replay-diverged", "dqn-replay", 3, "replay, no target network (seed 3)"),
    ("healthy", "dqn", 3, "replay + target network (seed 3)"),
    ("egocentric", "dqn", 0, "egocentric.v1 (27 inputs)"),
    ("features", "dqn-features", 0, "features.v1 (11 inputs)"),
    ("grid", "dqn-grid", 0, "grid-flat.v1 (100 inputs)"),
]


def recording() -> dict:
    registry, metrics, _ = TelemetryStores.open()
    runs = {(r.config["arm"], r.config["rng_seed"]): r for r in experiment_runs(registry, EXPERIMENT)}
    out = {}
    for key, arm, seed, label in PICKS:
        run = runs[(arm, seed)]
        history = metrics.history(run.run_id)
        steps = [int((h.extras or {})["env_steps"]) for h in history]
        out[key] = {
            "run_id": run.run_id,
            "label": label,
            "score": [
                [s, round(h.held_out_score, 2)]
                for s, h in zip(steps, history, strict=True)
                if h.held_out_score is not None
            ],
            # %.4g keeps a diverged 1.3e10 and a healthy 2.63 alike readable
            "q_mean": [[s, float(f"{(h.extras or {})['q_mean']:.4g}")] for s, h in zip(steps, history, strict=True)],
        }
    return {"experiment": EXPERIMENT, "runs": out}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(recording(), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
