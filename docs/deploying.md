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

CI's `deploy` job (`.github/workflows/ci.yml`) deploys the site. The whole pipeline is in the repo, and its logs are
readable with `gh run view --log`. After `python`, `rust` and `frontend` pass on a push to `main`, it:
1. **uploads ONNX Runtime if a dependency upgrade brought a new version** (`build-static.mjs --stage-ort`, then
   `deploy_r2.py --ort-only`, which uploads only files the bucket lacks), so the new site never asks for files the data
   host doesn't have;
2. **builds the site** (`pnpm generate:static`). The build is strict here: it fails if the data host doesn't serve its
   ONNX Runtime version. It also runs the size check and the local-identifier check (paths only on a build server);
3. **deploys the Worker** (`wrangler deploy`, from `apps/frontend/wrangler.jsonc`).

It never runs for pull requests, so forks never see the secrets. A PR's own static build only *warns* about a missing
ONNX Runtime version (`ORT_CHECK=warn`), since the upload happens when it merges. A second merge waits for a running
deploy rather than interrupting it. The **deploy** badge in the README shows the latest production deployment.

**The data stays a local step.** It comes from the local stores, which CI can't reach. After training, run
`uv run redqueen publish site --wait` and then `uv run redqueen dev deploy --no-site`. If a frontend change needs new
data (a new route, a new field), upload the data before merging.

**Setup: four repository secrets.** Until they're set, the job skips with a notice.

1. **Create a Cloudflare API token for deploying the Worker.** Go to **My Profile → API Tokens → Create Token** and use
   the **Edit Cloudflare Workers** template. Set **Account Resources** to your account and **Zone Resources** to
   `mackinations.ca` only. If a deploy fails on the custom domain, add **Zone → DNS → Edit** for `mackinations.ca`.
2. **Store the secrets in GitHub**, in your own terminal (PowerShell). The first command prompts for the token; paste
   it there, not into a file or a chat. The other three copy from the environment variables you already set:

   ```powershell
   gh secret set CLOUDFLARE_API_TOKEN
   gh secret set CLOUDFLARE_ACCOUNT_ID --body "$env:R2_ACCOUNT_ID"
   gh secret set R2_ACCESS_KEY_ID --body "$env:R2_ACCESS_KEY_ID"
   gh secret set R2_SECRET_ACCESS_KEY --body "$env:R2_SECRET_ACCESS_KEY"
   ```

   The R2 pair is the same Object Read & Write upload token; CI only adds versioned ONNX Runtime files with it.
3. **Don't also connect Cloudflare's Git integration** (Workers Builds). It would deploy alongside this job, and inside
   Cloudflare's build Nitro picks a different preset unless it's pinned (`build-static.mjs` pins it). If it's
   connected, disconnect it under **Workers & Pages → redqueen → Settings → Build**.
4. **Optional:** in **Settings → Environments → production**, add protection rules (for example, required reviewers)
   to gate each deploy.

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
