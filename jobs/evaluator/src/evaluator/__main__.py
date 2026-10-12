"""Run one evaluation spec: `uv run python -m evaluator SPEC [--set key=value ...]` (see jobcore.cli).

jobs/evaluator/specs/ has one spec per protocol. Prints the result: a leaderboard protocol's summary (its records are
in the evaluation store), or a report.
"""

from typing import Any

from jobcore.cli import run_spec_command

from evaluator import evaluate


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run_spec_command("evaluator", "evaluate", evaluate, argv)


if __name__ == "__main__":
    main()
