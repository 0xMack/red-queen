"""Write a real tabular RL run as the Learn chapter's recorded fallback (docs/design/0010: every live demo has one).

When a reader's browser can't run WebAssembly, the Q-learning chapter shows this instead of training live: the run's
learning curve (held-out score on the monitor games against env steps) and the Q-table it ended with. Values are
rounded to 4 decimals -- plenty to pick the same moves and color the same cells -- to keep the file small. A table
carries no visit counts, so `initial_q` goes along: a row still at its starting values was never updated (equal values
alone don't say that -- in a trapped row every move dies, and all three converge to the same number).

  uv run python jobs/export_rl_recording.py RUN_ID --label "..."   # -> apps/frontend/app/data/recordings/q-learning-snake.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelpack import QTable, load_champion
from run_context import TelemetryStores

OUT = (
    Path(__file__).resolve().parents[1] / "apps" / "frontend" / "app" / "data" / "recordings" / "q-learning-snake.json"
)


def recording(run_id: str, label: str | None = None) -> dict:
    registry, metrics, artifacts = TelemetryStores.open()
    run = registry.get_run(run_id)
    history = metrics.history(run_id)
    table = load_champion(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    if not isinstance(table, QTable):
        raise SystemExit(f"run {run_id}'s champion isn't a Q-table")
    params = run.config.get("params") or {}
    described = ", ".join(f"{k} {v:g}" for k, v in sorted(params.items())) or "default parameters"
    return {
        "run_id": run_id,
        "label": label
        or f"{run.config['representation'].replace('_', '-')} on Snake ({described}), rng seed {run.config['rng_seed']}",
        "algorithm": run.config["representation"],
        "params": params,
        "initial_q": params.get("initial_q", 0.0),
        "curve": [
            [int((h.extras or {}).get("env_steps", 0)), round(h.held_out_score, 3)]
            for h in history
            if h.held_out_score is not None
        ],
        "values": [round(v, 4) for v in table.values],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a tabular RL run as the Learn chapter's recorded demo.")
    parser.add_argument("run_id")
    parser.add_argument("--label", help="how the chapter describes the run (default: built from its config)")
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(recording(args.run_id, args.label), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
