"""Run one training spec: `uv run python -m trainer SPEC [--set key=value ...]` (see jobcore.cli).

jobs/trainer/specs/ has one spec per algorithm at the original scripts' defaults. Prints the run id; the run itself is
in the data directory (telemetry.data_dir()) like any other.
"""

import sys

from jobcore.cli import run_spec_command

from trainer import train


def main(argv: list[str] | None = None) -> str:
    return run_spec_command("trainer", "train", train, argv)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
