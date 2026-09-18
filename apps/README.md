# apps

Front-end applications — primarily interactive visualizations of games/simulations and algorithm
internals (e.g. watching a population evolve, stepping through an agent's decisions). See
[../docs/design/0005-frontend-and-api-contracts.md](../docs/design/0005-frontend-and-api-contracts.md)
for the contract design and the reasoning behind keeping this as one app to start.

## Contents

- `frontend/` — the one Nuxt 4 app for now: a run list and a live run-detail view over
  `apis/backend`, with room to grow into game viewing/interaction (doc 0005 steps 4-7) as a set of
  additional pages/stores within the same app rather than a second one.
