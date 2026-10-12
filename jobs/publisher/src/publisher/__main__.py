"""Run one publish spec: `uv run python -m publisher SPEC [--set key=value ...]` (see jobcore.cli).

jobs/publisher/specs/ has one spec per target. Prints what was stored.
"""

from typing import Any

from jobcore.cli import run_spec_command

from publisher import publish


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run_spec_command("publisher", "publish", publish, argv)


if __name__ == "__main__":
    main()
