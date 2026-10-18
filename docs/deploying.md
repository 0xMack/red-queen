# Deploying redqueen.mackinations.ca

The public site has no server (docs/design/0019). Two hosts serve it:
- **The site**, at **redqueen.mackinations.ca**: a Cloudflare Worker with only static assets, no script. The Nuxt
  build is its assets, configured in `apps/frontend/wrangler.jsonc`.
- **The data**, at **redqueen-data.mackinations.ca**: Cloudflare R2, holding a static copy of the API's public reads,
  the model files and ONNX Runtime.

Training and everything else stays on your machine. Static assets on Workers and R2 storage under 10 GB with no
egress fees are both free, so at this size it costs nothing.

## One-time setup

`mackinations.ca` must be a zone on your Cloudflare account. Both custom domains need that.

1. **Log Wrangler in** in your own terminal. It opens your browser to approve access:

   ```bash
   pnpm --dir apps/frontend dlx wrangler@4 login
   ```

2. **Run the setup.** It's safe to re-run: each step is skipped if it's already done.

   ```bash
   uv run redqueen dev cloudflare-setup --zone-id <mackinations.ca's zone ID>
   ```

   - It creates the `redqueen-data` bucket, disables its `r2.dev` URL, connects `redqueen-data.mackinations.ca`, and
     sets the CORS rule the site needs to read the bucket (the uploader's, `deploy_r2.py --print-cors`).
   - The zone ID is on the `mackinations.ca` Overview page, in the right-hand **API** section. It's only needed until
     the domain is connected.
   - The site's Worker needs no setup: the first `redqueen dev deploy` creates it and connects
     `redqueen.mackinations.ca`, including the DNS record and certificate.

3. **Create the R2 upload token in the dashboard**, the one step Wrangler can't do. Go to **R2 Object Storage →
   Manage API tokens → Create API token**, choose permission **Object Read & Write** (not "Object Read": that can't
   upload), and apply it to the `redqueen-data` bucket only. It can't change bucket settings, and that's deliberate:
   the credentials on your machine can only write objects.

   Set the values as user environment variables (PowerShell; the Account ID is in the dashboard URL,
   `dash.cloudflare.com/<account id>/...`), then restart whatever needs to see them, such as your terminal or the
   Claude desktop app:

   ```powershell
   [Environment]::SetEnvironmentVariable("R2_ACCOUNT_ID", "...", "User")
   [Environment]::SetEnvironmentVariable("R2_ACCESS_KEY_ID", "...", "User")
   [Environment]::SetEnvironmentVariable("R2_SECRET_ACCESS_KEY", "...", "User")
   ```

## Releasing

After training, and after re-ranking whatever changed (`uv run redqueen evaluate snake --wait` and so on), with the
API and scheduler running (`uv run redqueen dev up`):

```bash
uv run redqueen publish site --wait     # the snapshot: <data dir>/site (about 15 s)
uv run redqueen dev deploy --dry-run    # optional: what would upload, and a check of the Worker config
uv run redqueen dev deploy --no-site    # upload the data (the site deploys itself on merge, below)
```

What `dev deploy` does:
- **Uploads only what changed.** The bucket keeps the last upload's `index.json`, so a new run is a handful of files.
  Model files and ONNX Runtime are content- or version-named and cached for a year; API files are cached for five
  minutes, so a new snapshot shows within that.
- **Has partial forms.** `--no-site` uploads only data (new results; the app is unchanged). `--no-data` redeploys only
  the app (a frontend change). `--prune` also deletes bucket objects the new snapshot no longer lists.

## Automatic deploys (the site, on every merge to `main`)

Cloudflare's Git integration (Workers Builds) deploys the site itself. Its settings are in **Workers & Pages →
redqueen → Settings → Build**:

| Setting | Value |
|---|---|
| Repository / production branch | `0xMack/red-queen`, `main` |
| Root directory | `apps/frontend` |
| Build command | `pnpm install --frozen-lockfile && pnpm generate:static` |
| Deploy command | `npx wrangler deploy` (reads `wrangler.jsonc`) |
| Build watch paths | `apps/frontend/*` (a merge that doesn't touch the frontend doesn't rebuild) |

- **Versions are pinned** (`apps/frontend/.node-version` and `package.json`'s `packageManager`), so Cloudflare's build
  and CI's use the same Node and pnpm. The static build's checks run in both: file sizes, local identifiers (paths
  only on a build server), and ONNX Runtime (below).
- **Upgrading `onnxruntime-web`?** The site loads ONNX Runtime from the data host at `ort/<version>/`, and
  `pnpm generate:static` fails if that version isn't there. That stops Cloudflare from deploying a site whose models
  can't load, and makes the upgrade's PR fail CI. Upload the new version from your machine (after a local
  `pnpm generate:static`), then re-run the PR's checks:

  ```bash
  uv run --script jobs/publisher/scripts/deploy_r2.py --ort-only --ort apps/frontend/.output/ort
  ```

**The data stays a local step.** It comes from the local stores, which no build server can reach. After training, run
`uv run redqueen publish site --wait` and then `uv run redqueen dev deploy --no-site`. If a frontend change needs new
data (a new route, a new field), upload the data before merging, so the new site never asks for files that aren't
there yet.

## What gets published

- **Data:** the API's public reads (runs and their configs, curves, champions, leaderboards, catalogs) and the model
  files. Timestamps (when each run trained) are part of that.
- **Never:** live endpoints, jobs or experiments, unreferenced champions, and nothing about this machine. Training
  cost's `hardware` block is reduced to `engine` plus "one desktop computer".
- **The guard:** both `redqueen publish site` and `pnpm generate:static` fail, naming the file, if anything they'd
  publish contains this machine's home directory, username, hostname, repo path or CPU name.

## Checking it

- **https://redqueen.mackinations.ca/games/snake.** A champion should play, and the browser console should report
  `crossOriginIsolated === true`.
- **If uploads fail with `AccessDenied` on `PutObject`**, the token is read-only. Create an **Object Read & Write**
  one (step 3).
- **If models don't load**, open the browser's network tab. A blocked request to `redqueen-data.mackinations.ca`
  means the CORS rule is missing or doesn't list the site. Re-run `redqueen dev cloudflare-setup`, which sets it
  again.
- **If a deep link** (`/runs/<id>`) shows a 404, check `"not_found_handling": "single-page-application"` in
  `apps/frontend/wrangler.jsonc`.
