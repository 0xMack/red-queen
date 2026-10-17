# Deploying redqueen.mackinations.ca

The public site has no server (docs/design/0019). Cloudflare Pages serves the app at **redqueen.mackinations.ca**.
Cloudflare R2 serves the data at **redqueen-data.mackinations.ca**: a static copy of the API's public reads, the model
files, and ONNX Runtime. Training and everything else stays on your machine. On Cloudflare's free tiers this costs
nothing at this size: about 150 MB in R2, with no egress fees.

## One-time setup

You need a Cloudflare account with `mackinations.ca` added as a zone. If its DNS is managed elsewhere, move the
nameservers to Cloudflare first; custom domains on R2 and Pages need the zone on Cloudflare.

### 1. The data bucket

1. **Create the bucket.** Go to **R2 → Create bucket**, name it `redqueen-data`, and use the automatic location.
2. **Connect the domain.** In the bucket's **Settings → Custom Domains → Connect Domain**, enter
   `redqueen-data.mackinations.ca`. Cloudflare creates the DNS record and certificate. Use this hyphenated name:
   the free Universal SSL certificate covers `*.mackinations.ca` but not `data.redqueen.mackinations.ca`.
3. **Leave the `r2.dev` public URL disabled.** The custom domain is the only public way in.
4. **Create an API token.** Go to **R2 → Manage R2 API Tokens → Create API token**, with permission **Object Read &
   Write**, limited to the `redqueen-data` bucket. Note the **Access Key ID**, the **Secret Access Key**, and your
   **Account ID** (shown on the R2 overview).
5. **Set the CORS rule, once, in the dashboard.** The page is cross-origin isolated, so the bucket must allow the
   site to read it. Print the policy:

   ```bash
   uv run --script jobs/publisher/scripts/deploy_r2.py --print-cors
   ```

   Then go to **R2 → redqueen-data → Settings → CORS Policy → Add CORS policy**, paste the JSON, and save. It allows
   `GET`/`HEAD` from `https://redqueen.mackinations.ca` and the local dev and test ports.

   The upload token from step 4 can't do this itself. An **Object Read & Write** token can read and write objects but
   not change bucket settings, and that limit is deliberate: the credentials on your machine can't reconfigure the
   bucket. (`--configure-cors` applies the rule from the command line, but only with an **Admin Read & Write**
   token.)

### 2. The Pages project

1. **Authenticate Wrangler**, either once interactively:

   ```bash
   pnpm dlx wrangler@4 login
   ```

   or with a Cloudflare API token that has **Account → Cloudflare Pages → Edit** permission, set as
   `CLOUDFLARE_API_TOKEN` alongside `CLOUDFLARE_ACCOUNT_ID`.
2. **Create the project:**

   ```bash
   pnpm dlx wrangler@4 pages project create redqueen --production-branch main
   ```

3. **Add the domain.** In **Workers & Pages → redqueen → Custom domains → Set up a custom domain**, enter
   `redqueen.mackinations.ca`.

The site is deployed from your machine (direct upload), not built by Pages from git: its data comes from your local
runs anyway, and one command does both.

### 3. Credentials, in your environment (never in the repo)

```
R2_ACCOUNT_ID=...            # the Account ID
R2_ACCESS_KEY_ID=...         # the R2 token's Access Key ID
R2_SECRET_ACCESS_KEY=...     # the R2 token's Secret Access Key
CLOUDFLARE_ACCOUNT_ID=...    # the same Account ID (for Wrangler; or use `wrangler login`)
CLOUDFLARE_API_TOKEN=...     # optional, instead of `wrangler login`
```

On Windows, set them as user environment variables (System Properties → Environment Variables), or in your shell
profile.

## Releasing

After training, and after re-ranking whatever changed (`uv run redqueen evaluate snake --wait` and so on), with the
API and scheduler running (`uv run redqueen dev up`):

```bash
uv run redqueen publish site --wait     # the snapshot: <data dir>/site (about 15 s)
uv run redqueen dev deploy --dry-run    # optional: what would upload
uv run redqueen dev deploy              # build the site, upload the data, deploy the site
```

What `dev deploy` does:
- **Uploads only what changed.** The bucket keeps the last upload's `index.json`, so a new run is a handful of files.
  Model files and ONNX Runtime are content- or version-named and cached for a year; API files are cached for five
  minutes, so a new snapshot shows within that.
- **Has partial forms.** `--no-site` uploads only data (new results; the app is unchanged). `--no-data` redeploys only
  the app (a frontend change). `--prune` also deletes bucket objects the new snapshot no longer lists.

## Checking it

- **https://redqueen.mackinations.ca/games/snake.** A champion should play, and the browser console should report
  `crossOriginIsolated === true`.
- **If models don't load**, open the browser's network tab. A blocked request to `redqueen-data.mackinations.ca`
  means the CORS rule is missing or doesn't list the site: check **R2 → redqueen-data → Settings → CORS Policy**
  against `deploy_r2.py --print-cors`.
- **If a deep link** (`/runs/<id>`) returns Pages' 404 page instead of the app, check that `404.html` didn't end up in
  `.output/public`. `pnpm generate:static` removes it.
