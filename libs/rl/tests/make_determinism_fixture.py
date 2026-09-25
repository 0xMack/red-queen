"""Writes determinism.json: digests of fixed training computations, computed by the native build (docs/design/0010
Decision 2). The native build (test_determinism.py), the WASM build in Node
(apps/frontend/scripts/check-rl-determinism.mjs) and in a browser (/dev/rl) must all reproduce them.

Regenerate only for an intended change to the arithmetic: uv run python libs/rl/tests/make_determinism_fixture.py
"""

import json
from pathlib import Path

from rl import _native

FIXTURE = Path(__file__).with_name("determinism.json")
TRAINING = [(0, 200), (1, 50), (7, 500)]  # (seed, updates)
ROLLOUTS = [0, 1, 42]
LEARNING = [0, 5]
DQN = [0, 3]


def digests() -> dict:
    return {
        "training": [{"seed": s, "updates": u, "digest": _native.training_digest(s, u)} for s, u in TRAINING],
        "rollouts": [{"seed": s, "digest": _native.rollout_digest(s)} for s in ROLLOUTS],
        "learning": [{"seed": s, "digest": _native.learning_digest(s)} for s in LEARNING],
        "dqn": [{"seed": s, "digest": _native.dqn_digest(s)} for s in DQN],
    }


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(digests(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE}")
