import sys
from pathlib import Path

# jobs/ isn't a uv workspace package (see AGENTS.md) -- baseline_gp_run.py can import control.py
# because `uv run python jobs/<script>.py` puts the script's own directory on sys.path
# automatically. Tests need the same thing done explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
