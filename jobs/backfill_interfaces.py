"""One-off migration (docs/design/0007): record `config.interface` on game runs recorded before
interfaces existed.

Each legacy run is resolved from its own champion artifact's layer sizes (via
games.interfaces.find), not assumed -- the pre-0007 Snake runs were trained against two different
observations (a 100-float grid, then 11 features), and a champion only runs correctly under the one
it was trained for. Also records `training_seeds` (games.snake.BENCHMARK_SEEDS, which every legacy
Snake run trained on) so leaderboards can show the train-vs-held-out gap.

Writes straight to the local SqliteRunRegistry's `runs` table: config is otherwise immutable after
create_run(), and a registry-protocol method just for a one-time backfill isn't worth adding.
Idempotent -- runs that already have an interface are left alone.

Run with: uv run python jobs/backfill_interfaces.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from evolve.neuro import WeightVector
from games import interfaces
from games.snake import BENCHMARK_SEEDS
from run_context import RUN_DATA_DIR
from telemetry import FileArtifactStore, FileMetricsStore


def resolve_interface(run_id: str, game: str, metrics: FileMetricsStore, artifacts: FileArtifactStore) -> str | None:
    history = metrics.history(run_id)
    if not history:
        return None
    champion = WeightVector.from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    match = interfaces.find(game, input_size=champion.layer_sizes[0], num_outputs=champion.layer_sizes[-1])
    return match.id if match else None


def backfill(run_data_dir: Path) -> list[tuple[str, str | None]]:
    metrics = FileMetricsStore(run_data_dir / "metrics")
    artifacts = FileArtifactStore(run_data_dir / "artifacts")
    conn = sqlite3.connect(run_data_dir / "runs.db")
    changed: list[tuple[str, str | None]] = []
    try:
        for run_id, config_json in conn.execute("SELECT run_id, config FROM runs").fetchall():
            config = json.loads(config_json)
            game = config.get("game")
            if not game or "interface" in config:
                continue
            interface_id = resolve_interface(run_id, game, metrics, artifacts)
            changed.append((run_id, interface_id))
            if interface_id is None:
                continue
            config["interface"] = interface_id
            config.setdefault("training_seeds", list(BENCHMARK_SEEDS))
            conn.execute("UPDATE runs SET config = ? WHERE run_id = ?", (json.dumps(config), run_id))
        conn.commit()
    finally:
        conn.close()
    return changed


def main() -> None:
    for run_id, interface_id in backfill(RUN_DATA_DIR):
        print(f"{run_id}: {interface_id or 'no champion / no matching interface -- left unchanged'}")


if __name__ == "__main__":
    main()
