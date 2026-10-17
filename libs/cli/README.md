# redqueen-cli

The `redqueen` command (docs/design/0018, decisions 1, 5 and 8): the backend's API from the command line. An HTTP
client only, with no jobcore, no games and nothing native. The API validates every spec and refuses one that can't run
(422, with the reason), so the CLI runs anywhere the API is reachable. Its import name is `redqueen_cli`, because
`redqueen` is RedQueenCbind's C++ extension.

```bash
uv run redqueen --help
uv run redqueen runs list --game checkers
uv run redqueen train td_lambda --game checkers --budget iterations=200 -p agent.hidden=64 --wait
uv run redqueen evaluate checkers --wait                     # re-rank the leaderboard
uv run redqueen h2h material-4 3a9f1c                        # an SPRT, run-id prefixes resolve
uv run redqueen experiments submit experiments/selfplay-v8.yaml
uv run redqueen experiments status selfplay-v8
uv run redqueen pipeline submit experiments/pipelines/checkers-recipe.yaml --wait
uv run redqueen schedules submit experiments/schedules/checkers-rerank.yaml
uv run redqueen jobs logs 1f2e -f
uv run redqueen doctor
```

The groups follow the API's schema slices:

| Group | Commands | API |
|---|---|---|
| `runs` | `list`, `show`, `history`, `tail`, `pause`, `resume`, `step`, `artifact`, `brain` | `/runs` |
| `jobs` | `submit`, `list`, `show`, `logs [-f]`, `cancel` | `/jobs` |
| `experiments` | `submit`, `list`, `status`, `report [--h2h] [--full]` | `/experiments` |
| `pipeline` | `submit [--wait]` | `/jobs` (kind `pipeline`) |
| `schedules` | `submit`, `list`, `show`, `delete` | `/schedules` |
| `leaderboard` | `show GAME [--protocol]`, `interfaces GAME` | `/games/{game}/leaderboard` |
| `models` | `catalog GAME`, `package RUN REF` | `/games/{game}/models`, `/runs/.../package` |

The verbs `train`, `evaluate`, `h2h` and `publish` build one spec from options and submit it. `schemas` lists what the
API accepts, and `doctor` checks the setup.

Two groups produce files in the repo:

- **`export`** (`ballot`, `pbt-history`, `rl-curves`, `rl-recording`) regenerates the Learn chapters' data files. Each
  command submits a publisher job (`frontend.*` targets), so it still goes through the API, and the file is written
  in the checkout the scheduler runs from.
- **`dev`** is the only local-only group. `up` starts the backend, the scheduler and the frontend together, and
  Ctrl-C stops them cleanly. `deploy` publishes redqueen.mackinations.ca (docs/deploying.md). `build-wasm [games|rl]`, `openapi`,
  `bench-rl` and `screenshots RUN` run the repo's own scripts as subprocesses; the CLI still imports none of them.

**`redqueen mcp serve`** serves the same API as MCP tools for agents. Install it with
`uv sync --all-packages --all-extras` (it is the optional `mcp` extra), then add it to an agent, for example
`claude mcp add redqueen -- uv run redqueen mcp serve`. There are 22 tools: runs, leaderboards and models to read;
`train`, `evaluate`, `publish`, `submit_job`, `submit_spec_file`, experiments and schedules to start work. Long work
is never one blocking call. You submit, then poll `get_job` / `job_logs` (pass the returned offset back to follow a
log), and stop it with `cancel_job`. When the API refuses a request, the tool error carries the API's reason.

Every command takes `--json`, which prints the API's JSON unchanged in stable shapes, for scripts and the coming MCP
server. Every command that starts work takes `--wait`: it follows the job's log (a pipeline's, stage by stage) until
the job ends, and exits 1 unless it succeeded. `--api URL` (or `REDQUEEN_API_URL`) selects the API; the default is
`http://localhost:8000`. Nothing runs without the scheduler (`uv run python -m scheduler`). The API only records what
is asked for.

**`rq`** is installed as an alias, but nothing depends on it. RQ (Redis Queue) also installs an `rq`, and outside the
venv, PATH order picks the winner silently. Docs, CI and scripts always call `redqueen`. `redqueen doctor` reports what
`rq` resolves to, and `tests/test_entry_points.py` fails if any workspace dependency ships an `rq` script of its own.

Tests (`uv run pytest libs/cli`) drive the real backend through its `TestClient`, so every command is exercised
against the actual API.
