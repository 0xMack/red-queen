# apis

Backend APIs serving runs/simulations to `apps/`. See
[../docs/design/0005-frontend-and-api-contracts.md](../docs/design/0005-frontend-and-api-contracts.md)
for the contract design and the reasoning behind keeping this as one service to start.

## Contents

- `backend/` — the one FastAPI service for now: run/metrics/artifact endpoints today
  (`routers/runs.py`), games endpoints to come (`routers/games.py`, doc 0005 step 4). Kept as a
  single deployable with routers grouped by concern rather than split into multiple services, so a
  future split (if one is ever needed) is a mechanical extraction, not a redesign.
