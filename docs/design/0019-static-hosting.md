# 0019 — Hosting the site with no server

Status: **Accepted; built** (2026-10-17): the snapshot (`site` target), the frontend's static mode
(`pnpm generate:static`) and `redqueen dev deploy`. The one-time Cloudflare setup and the release loop are in
[docs/deploying.md](../deploying.md).
Relates to: [0005](0005-frontend-and-api-contracts.md) (one app, one API; `useApi` is the only door to the backend),
[0009](0009-client-side-inference-at-scale.md) (everything a visitor runs, runs in their browser; model packages are
content-addressed blobs, and whether R2 or the HF Hub hosts them was left open),
[0018](0018-workloads-api-and-cli.md) (workloads, publish targets, the CLI).

## Context

The site is going public at **redqueen.mackinations.ca**, at close to zero cost. Training stays on the local machine.
A visitor's games, models and Learn labs already run in their browser (0009). Everything else they see is a GET of
data that only changes when training or evaluation runs locally: runs, curves, champions, leaderboards, catalogs.
The API also has no authentication (0018 deferred it), so its job, experiment, schedule and write routes must never
be reachable from the internet.

## Decision

**Publish a static copy of the API's public reads; run no server.**

1. **The snapshot is the API rendering itself.** `backend.static_site.export` (the publisher's `site` target,
   `redqueen publish site`) fetches every GET route the frontend reads through the app in-process (a `TestClient`
   over the local stores). It writes each response at its own path plus `.json`, so
   `/runs/{id}/metrics/history` becomes `api/runs/{id}/metrics/history.json`. A file is what the API answered, and
   `jobs/publisher/tests/test_site.py` checks every file against the live route, so the copy can't drift from the
   API. A second implementation of the routes would drift.
2. **Done ahead of time:** the on-demand package export (`POST .../package`), for each Snake run's latest champion.
   The response is stored as `.../package.json`.
3. **Left out:**
   - runs still `running`/`paused`, which would look frozen;
   - every artifact except the champions a page asks for (each run's latest, and every leaderboard champion), since
     the artifact store is gigabytes of per-generation snapshots;
   - model files no catalog or package refers to, and the `scale-test` catalog (~340 MB, for the unlinked
     `/dev/inference`);
   - everything live: the metrics stream, run control, jobs, experiments, schedules, the write endpoints.
4. **Hosting:**
   - **A Cloudflare Worker with only static assets** serves the Nuxt app at `redqueen.mackinations.ca`
     (`apps/frontend/wrangler.jsonc`: no script, single-page fallback, the custom domain). This was Pages at first;
     Wrangler now routes Pages to Workers, where the whole setup, domain included, is scriptable.
   - **Cloudflare R2** serves the snapshot and the model files at `redqueen-data.mackinations.ca`, because R2 has
     no egress fees. Catalogs' `base_url` is `https://redqueen-data.mackinations.ca/models`.
   - The data host is `redqueen-data.`, not `data.redqueen.`, because Cloudflare's free Universal SSL certificate
     covers only one level of subdomain.
   - Data and site deploy separately: a new training result is a data upload, not a site rebuild.
5. **Nothing about the machine is published.** The training-cost meter's `hardware` block (CPU model, cores, OS, Python
   version) is reduced to its `engine` and a generic `hardware_class` in the copy -- the only change to an API answer;
   local records keep it. Both builds then refuse to keep a file containing one of this machine's identifiers (home
   directory, username, hostname, the repo's path, the CPU's name -- read from the machine, never written in code):
   the snapshot (`PrivacyError`, `backend.static_site.check_private`) and `pnpm generate:static` (which caught the Rust
   WASM carrying its absolute source paths; the WASM builds now remap them).
6. **Incremental uploads:** `index.json` lists every file with its sha256, and a deploy uploads only the files whose
   hash differs from the bucket's copy of the index. Model blobs are named by their hash, so they never re-upload.
   The uploader (`jobs/publisher/scripts/deploy_r2.py`, R2's S3 API) is a standalone `uv run --script` with its own
   boto3, so no workspace package depends on it. The site is deployed from this machine (`wrangler deploy`), not built
   from git, because its data is local anyway. `redqueen dev cloudflare-setup` does the rest of the setup through
   Wrangler, except the R2 upload token. `redqueen dev deploy` runs the three steps in order -- build, upload data,
   deploy site -- so a new site never asks for files that aren't there yet.

## Consequences

- The public site has no live training and no pinned old generations (only each run's latest champion is copied).
  Both remain available locally.
- The frontend needs a static mode. `useApi` maps a path to its `.json` file and refuses non-GETs; `/watch` reads
  the pre-built package; the metrics store doesn't open a stream; run control is hidden.
- ONNX Runtime's own `.mjs`/`.wasm` are served from the data host, not the site: its WebGPU build (~26 MB) is over
  Cloudflare's 25 MiB per-asset limit. The static build fails if any file is over the limit.
- The page must be cross-origin isolated (`Cross-Origin-Opener-Policy`/`-Embedder-Policy`) for multithreaded ONNX
  Runtime. The data host must then send CORS headers, and third-party resources the page loads (Google Fonts today)
  must allow it, or be served from the site itself.
- The release loop is all local: train, re-rank, `redqueen publish site`, deploy.
