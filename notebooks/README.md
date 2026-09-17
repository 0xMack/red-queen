# notebooks

Jupyter notebooks for exploration: algorithm experiments, analysis, and write-ups. Per
docs/design/0003, this is the first home for comparing techniques — raw runs live in `telemetry`;
a notebook is the narrative on top explaining what a comparison shows.

## Contents

- `0001-tournament-vs-lexicase.ipynb` — the first comparison notebook (docs/design/0003 phase 2):
  `TournamentSelection` vs `LexicaseSelection` on the fixed benchmark from
  `jobs/baseline_gp_run.py`, plus an isolated demo of why lexicase can prefer a "specialist" over a
  higher-mean "generalist". Re-run in place with
  `uv run jupyter execute --inplace notebooks/0001-tournament-vs-lexicase.ipynb`.
