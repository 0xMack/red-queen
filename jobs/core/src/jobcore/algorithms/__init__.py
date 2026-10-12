"""Every built-in algorithm's params model and registration (docs/design/0018, decision 1). Importing this registers
them all; `jobcore.specs` does so on first use. The defaults are the original scripts' defaults, so a spec that sets
nothing trains exactly what the script did."""

from jobcore.algorithms import evolution, language, reinforcement  # noqa: F401
