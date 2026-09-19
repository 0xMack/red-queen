# 0009 — Client-side inference at scale

Status: **Implemented through step 6** (see "Implementation notes" at the end for what changed on the
way, measured results, and what's left: hosting (step 7), recorded episodes (step 8), and the pixel
conv policy of step 5).
Relates to: [0005](0005-frontend-and-api-contracts.md) (supersedes its "WebAssembly for game rendering —
decided: Pyodide" section for game simulation, and its interaction modes 2/3 for how a policy runs),
[0007](0007-representations-leaderboards-and-tradeoffs.md) (interfaces, evaluation protocols, inference-cost
metrics — all reused), [0008](0008-neat-and-tracked-comparisons.md) (NEAT wire format and the TS port),
[0004](0004-small-transformer-from-scratch.md) (`tinylm`, the first model that won't fit today's path).

## Context

This project trains models server-side. Visitors watch the trained models play in their own browser. Once the site is
hosted, **visitor compute costs nothing and visitor bytes cost money**. Every forward pass run in a visitor's
browser is one the server doesn't pay for. Every megabyte of weights they download is egress that *someone* pays for,
unless the storage has none. This doc is about moving as much inference as possible into the browser, for
models much larger than today's, without making hosting costs grow with the number of visitors.

### What today's path is, and where it stops

A champion is fetched from `GET /runs/{id}/artifacts/{ref}` (served by Python). It is a JSON `WeightVector` or `NeatGenome`,
loaded into Pyodide alongside the real `libs/games` + `libs/evolve` source, and `forward()` runs as pure Python
inside `app/workers/snakeGame.worker.ts`. That was the right call for a 243-weight MLP (doc 0005: one source of
truth, zero divergence). It does not scale to conv nets, image models, or transformers:

| Limit | Today | Why it breaks at scale |
|---|---|---|
| Compute | pure-Python `forward()` in CPython-on-WASM, float64, no SIMD, no GPU | orders of magnitude slower than a real kernel. A conv layer or attention block in pure Python is unusable |
| Weight format | JSON floats | ~3× larger than float32 binary, can't be quantized, can't stream, must all be parsed into one heap |
| Delivery | through FastAPI, every time, no client cache | every visitor download of a 500 MB model costs Python CPU plus egress, and repeat visits download it again |
| Runtime startup | Pyodide cold load ~11 s, several MB, before anything plays | acceptable for one demo, and it runs *before* the model download, which is now the long pole |
| Device fit | assumed, never checked | a 1 GB model on a phone either crashes the tab or never finishes, with no explanation |
| Parity | same Python both sides, so exact by construction | a GPU runtime in float16/int4 is *not* exact, so we need a measured parity guarantee instead of a structural one |

## Goals and non-goals

- **Goal:** any model we train, up to **~1 GB quantized**, can be watched in a capable visitor's browser, and a
  visitor's device never costs server compute per frame, per decision, or per token.
- **Goal:** a visitor whose device can't run a model sees *that it can't and why*, and can still watch every model
  their device *can* run.
- **Goal:** the framework is complete before hosting is decided. Storage provider, CDN, and domain are
  configuration, not architecture.
- **Non-goal (for now):** visitor-facing server-side inference. The server does *offline, batch* inference only
  (evaluation, leaderboards). A quota'd opt-in endpoint may come later; this doc leaves a seam for it and builds
  nothing (Decision 7).
- **Non-goal:** training in the browser. The existing Learn demos (TS NEAT, TS ES) are teaching tools, not a
  training path.

## Decision 1: split the *game runtime* from the *model runtime*

Today one Pyodide process does both jobs: stepping the game and running the policy. They have opposite needs:

- **The game** is small, branchy, integer-heavy, and must be **bit-identical** between the Python used for
  training/evaluation and the browser, because a leaderboard score is only meaningful if the visitor watches the same game.
- **The model** is large, dense, float-heavy, wants a GPU, and tolerates bounded numeric error as long as that error is
  *measured* (Decision 4).

So they become two runtimes that share nothing but a typed-array boundary:

```
          ┌──────────────── session worker (one per watched game) ────────────────┐
 main  ◄──│  game runtime (Rust→WASM)          model runtime (ORT Web)            │
 thread   │   reset(seed) / step(action)  ──►   observation: Float32Array  ──►     │
 (render  │   observer.encode()                 session.run()                     │
  only)   │   adapter.decode(outputs)     ◄──   outputs: Float32Array             │
          └───────────────────────────────────────────────────────────────────────┘
```

Both live in the same dedicated worker (WebGPU is available in workers), so a decision costs no `postMessage`.
The main thread only receives render states, as today.

## Decision 2: the game core moves to Rust, compiled for both Python and WASM

**Decided (per review):** game logic, observers, action adapters, and baselines move from `libs/games` Python
into one Rust core. It is bound to Python with **PyO3/maturin** for training and evaluation, and compiled to
**WASM with wasm-bindgen** for the browser. This supersedes doc 0005's Pyodide decision *for games*. Doc 0005's
underlying argument ("one implementation, run on both sides, so they can't diverge") still holds. Rust keeps that
property and loses Pyodide's ~11 s / multi-MB startup and the per-tick Python↔JS crossing an ORT-backed policy
would otherwise pay.

- **Why Rust over C++/Emscripten** (the `RedQueenCbind` precedent): wasm-bindgen emits a small module with no
  Emscripten runtime; maturin is a first-class PEP 517 backend, so the package stays a normal `uv` workspace member;
  and games are exactly the stateful, index-heavy code where memory safety pays off. `RedQueenCbind` stays C++.
  Nothing here touches it.
- **Layout.** One core for every game (one package for all games, not one per game, as today):

  ```
  libs/games/
    Cargo.toml            cargo workspace: core, python, wasm
    core/                 pure Rust: Snake, Checkers, Reach1d, observers, adapters, baselines, interfaces
    python/               PyO3 bindings → the importable `games` package (pyproject.toml, maturin backend)
    wasm/                 wasm-bindgen bindings → built into apps/frontend by `wasm-pack`
  ```

  The Python import surface (`games.snake.Snake(observer=...)`, `games.interfaces.get(...)`,
  `render_state()`, `explain()`) stays the same, so `evolve`, `jobs/`, and `apis/backend` don't change. PyO3 classes
  satisfy `evolve`'s structural `Environment` interface the same way the Python classes do today. Training
  also gets a faster simulator as a side effect.
- **Determinism is specified, not inherited.** Games currently seed Python's `random` (Mersenne Twister). The Rust
  core carries its own small, documented PRNG (e.g. PCG32 with a spelled-out seeding rule), so any future
  port can reproduce it. Continuous games (`reach1d`) use the `libm` crate on *both* targets. Basic IEEE
  arithmetic is deterministic, but platform `sin`/`exp` are not.
- **Consequence for leaderboards:** seed *N* produces a different Snake layout under the new PRNG, so held-out
  scores change. Per doc 0007 that is a **protocol version bump** (`snake.score.v2`), not a silent edit: every
  entrant is re-evaluated under v2. Champions themselves keep working, because an interface's *encoding* is unchanged
  (`features.v1` is still `features.v1`). The Rust observer must match the Python one bit for bit, and that is tested.
- **Migration is oracle-checked.** Until parity is shown, the existing Python implementation stays as a test
  oracle. Given the same PRNG stream (injected), both must produce identical trajectories, observations,
  `explain()` output, and render states over thousands of random-action episodes. Only then is the Python deleted.
  (This is doc 0008's "check the port against the Python" discipline, applied before the switch rather than after.)
- **Pyodide retires.** Once every game is ported, nothing the browser needs is Python, so the Pyodide
  worker and the `server/api/py-source/[pkg]` route go away. That also lifts `evolve`'s "pure stdlib, so it loads
  in Pyodide" constraint. It can take numpy if it ever needs it.

## Decision 3: the model runtime is ONNX Runtime Web, with hand-rolled kernels only where they teach

**Decided (per review): hybrid.** Production inference uses **ONNX Runtime Web** behind one small interface of our
own. Hand-written kernels exist where seeing the mechanism *is* the point (Learn pages), not on the production path.

```ts
interface ModelRuntime {
  load(pkg: ModelPackage, variant: VariantId, onProgress): Promise<LoadedModel>
}
interface LoadedModel {
  run(inputs: Record<string, TypedArray>): Promise<Record<string, TypedArray>>
  release(): Promise<void>
}
```

Backends behind it:

| Backend | Used for | Notes |
|---|---|---|
| `ort-webgpu` | anything conv/attention-sized; required for large models | WebGPU EP (the WebGL/JSEP paths are deprecated upstream). Weights stay GPU-resident; for autoregressive models, KV-cache stays on GPU via `preferredOutputLocation: "gpu-buffer"` (IO binding), so no per-token readback |
| `ort-wasm` | small/medium models; fallback where WebGPU is missing | SIMD always; multithreaded only when the page is cross-origin isolated (see below). The WASM32 heap tops out at 4 GB, so large models are WebGPU-only in practice |
| `native-ts` | Learn chapters; per-neuron activation visualization of tiny nets | today's `utils/neat.ts`/`snakePolicy.ts` role, formalized. Never the source of a leaderboard number |

- **Why ORT and not a from-scratch WebGPU engine:** 1 GB int4 transformers need fused, tuned kernels (quantized
  matmul, attention, layer norm) on four GPU vendors. That is a project the size of everything else here combined. ORT
  already ships them (e.g. `MatMulNBits` for int4 on WebGPU), and ONNX is also the natural export target for
  PyTorch. A hand-written WebGPU matmul/conv chapter on `/learn` is still worth doing *as a lesson*.
- **Why not WebLLM/MLC or transformers.js:** WebLLM is LLM-only and needs a TVM compile toolchain, so it would be a second
  runtime. transformers.js is itself ORT underneath plus HF tokenizers and pipelines. We'd rather own the generation loop
  (sampling, KV-cache) for `tinylm` and use ORT directly. Revisit transformers.js if we start shipping third-party
  HF models.
- **WebNN** (ORT has an EP for it) is watched, not adopted. It would add NPU access, but support isn't broad enough yet.
- **Cross-origin isolation:** multithreaded WASM needs `SharedArrayBuffer`, which needs the app served with
  `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`).
  Every cross-origin subresource (weights on a CDN) then needs CORS/CORP headers. That becomes a
  **hosting requirement**, recorded now so it doesn't surprise the deploy. Without isolation, `ort-wasm` falls back to
  one thread and keeps working.

## Decision 4: one portable *model package* for every trainer, with measured parity

**Decided (per review): both trainers, one format.** Evolved and from-scratch models (`evolve`, `autodiff`,
`tinylm`) and, where scale matters, PyTorch-trained ones all export to the same artifact. The browser never knows
which trainer produced a model.

A **model package** is immutable and content-addressed:

```
manifest.json                   small, fetched first, decides everything below
model.onnx                      graph (weights external, so the graph stays small)
weights/<sha256>.bin …          external-data shards, ~32–64 MB each (streamable, resumable, cacheable)
parity fixture (JSON)           recorded inputs + reference outputs (export-time check; the in-browser self-test)
```

`manifest.json` (a pydantic model in Python, mirrored in `app/types/`):

- `format_version`, `package_id` (hash of everything else), `interface` (doc 0007 id, e.g.
  `snake/features.v1+relative3.v1`, or `lm/bytes.v1`). A model still only runs under its own interface.
- `provenance`: run id, `champion_ref`, trainer (`evolve.neat`, `autodiff`, `torch`), export tool versions.
- `io`: named inputs/outputs with dtypes and shapes (symbolic dims for sequence length).
- `variants`: e.g. `fp32`, `fp16`, `int8`, `int4`, each with its own shard list and byte total, plus
  **requirements**: allowed backends, WebGPU features (e.g. `shader-f16`), estimated peak GPU/heap memory, and
  `max_buffer_bytes` (its largest single tensor).
- `parity`: per variant, the max abs error vs. the reference forward, and **the action-agreement rate on the
  evaluation protocol's held-out games** (below).

### Exporters

A new leaf-ish package, **`libs/modelpack/`**: manifest schema, exporters, parity checking, publishing. It depends
on `onnx` and `onnxruntime` (Python), and optionally on `torch` as an extra. `evolve`/`autodiff` stay free of
ONNX. `modelpack` depends on them, not the other way round (same direction as `jobs/` → `telemetry`).

| Source | Export |
|---|---|
| `WeightVector` | `Gemm → Tanh` chain. Trivial |
| `NeatGenome` | the arbitrary DAG is compiled to *layers*: topologically group nodes, and each layer is one `MatMul` over the running concatenation `[inputs, bias, hidden-so-far]` with zeros where genes are absent, then `Tanh`, then `Concat`. A few dense ops instead of hundreds of scalar ones. Disabled genes are dropped and dead ends are pruned (the `cached_property` evaluation order `neat.py` already computes) |
| `autodiff` / `tinylm` | a small graph builder over `onnx.helper`, one mapping per op `autodiff` defines. This is the only exporter we write in depth |
| PyTorch | `torch.onnx.export` (dynamo), then ORT's quantization tooling for int8/int4 |
| `LinearProgram` (GP) | not exported. It isn't a game policy, and `utils/linearProgram.ts` already covers its visualization |

### Parity is a number on the manifest, not a hope

Today parity is exact by construction (same Python both sides). After this doc it isn't: ONNX `float32` differs
from `WeightVector`'s float64, and int4 differs a lot more. The failure that matters isn't numeric error. It's a
**different move**: an `argmax` near a tie flips, and the visitor watches a different game than the one the
leaderboard scored. So at export time:

1. Max abs error on recorded observations (`parity.npz`). A per-dtype tolerance gates the export.
2. **Action agreement:** play the protocol's held-out games with the reference forward and with each variant
   (ORT Python, same kernels family), and record the fraction of identical decisions and whether the score matched.
3. A variant with < 100% agreement is a **distinct leaderboard entrant** (`<champion>@int8`), evaluated in its own
   right, with its own score and cost row in doc 0007's table. This makes "what does int8 cost us?" a
   leaderboard question, which is the kind of tradeoff 0007 exists to show.

The same principle covers evaluation: `jobs/evaluate.py` runs *the package visitors download* (ORT Python, CPU),
not the in-memory Python object, so a leaderboard score describes exactly what is watched. Inference-cost
columns gain ORT CPU µs/decision (hardware-fingerprinted, as today), artifact bytes per variant, and a FLOPs
estimate, which is hardware-independent and so comparable across machines, where 0007 says clocks aren't.

(Browser GPU kernels are not bit-identical to ORT's CPU kernels either. For small policies the fp32 WASM
backend is close enough to CPU that agreement is expected to stay 100%, and that's checked by the optional
in-browser self-test. For large models "the leaderboard scored the CPU run of this variant" is the honest label.)

## Decision 5: delivery. Content-addressed, cached forever, host TBD

**Host is undecided (per review): Cloudflare R2 behind a CDN, or the Hugging Face Hub.** The design makes the choice
a config value:

- A **`ModelStore` protocol** in `modelpack` (Protocol-first, like `telemetry`'s stores): `publish(package) → base
  URL`, `exists(package_id)`. Implementations: `LocalModelStore` (a directory, served as static files by
  `apis/backend` in dev, not through Python per request), later `R2ModelStore` / `HuggingFaceModelStore`.
  R2's zero egress fees suit the "visitor bytes cost money" goal directly. The HF Hub is free public model hosting
  with CORS, but comes with its own rate limits and terms. Either works, because the client only ever sees URLs.
- **Immutable URLs, mutable catalog.** Package files never change (hash-named), so they're served with
  `Cache-Control: immutable` and cached indefinitely by CDNs and browsers. The only mutable thing is a small
  **catalog** per game (which published packages exist, and which is the leaderboard entrant), served by
  `apis/backend` (`GET /games/{game}/models`) or as a static JSON.
- **Publishing is explicit and rare.** Only leaderboard-bound champions get a package, not every generation's
  champion. Live "watch training" (0005 mode 3) keeps using `GET /runs/{id}/artifacts/{ref}` for small models,
  which is a dev/training-time feature. A large model under live training publishes every *N* generations at most.
- **Client-side caching:** shards are stored in the Cache API (or OPFS) keyed by hash, so a returning visitor
  downloads nothing and a hash can never be stale. Before a large download: `navigator.storage.estimate()`
  against the variant's byte total, and `navigator.storage.persist()` requested for models over ~100 MB so the browser
  doesn't evict them. Downloads stream per shard with progress, and resume at the shard boundary after a
  failure.
- **Bandwidth is the cost to protect**, so any variant over a threshold (start at ~50 MB) needs an explicit
  click ("Download 640 MB to watch this model?"). Nothing large auto-downloads on a page view.

## Decision 6: capability matching. "Unsupported, and here's why", never a crash

**Decided (per review):** a visitor sees every model. Those their device can't run are shown as unsupported *with the
reason*, and every model that can run remains watchable.

- **`probeDevice()`** (once per session, cached): WebGPU adapter present? Adapter limits
  (`maxBufferSize`, `maxStorageBufferBindingSize`), features (`shader-f16`), and vendor/architecture info. WASM SIMD,
  `crossOriginIsolated` (threads), `navigator.deviceMemory` (coarse hint, Chromium only), storage quota,
  mobile vs. desktop.
- **`matchVariant(manifest, profile)`** picks the best runnable variant (e.g. fp16-WebGPU > int8-WebGPU >
  int8-WASM) or returns **structured reasons** for each variant it rejected. The UI renders those reasons in plain
  language: *"Needs WebGPU. Firefox on Linux doesn't ship it yet; try Chrome."*, *"Needs a ~900 MB GPU buffer; this GPU
  allows 256 MB."*, *"1.2 GB download; this browser offers 600 MB of storage."*
- **Estimates can be wrong, so runtime failures use the same path.** Allocation failures, `GPUDevice` loss, and tab
  memory pressure during load or run are caught, converted into the same reason type, and remembered locally per
  (package, variant) so the device isn't asked to fail twice.
- **Leaderboards and game pages** always list every entrant. A "watch" affordance appears only where
  `matchVariant` succeeds, and a device filter ("show only what runs here") sits on the leaderboard.
- **Several models at once** (the `/games/{game}` race against every entrant) goes through a
  memory budget in the model runtime. Small models load together. Large ones load one at a time, with least-recently-used
  `release()`, so a race page can't stack four 500 MB models on a laptop GPU.

## Decision 7: server-side inference stays offline, with one seam for later

**Decided (per review):** the server runs models only in batch jobs, evaluation and leaderboards (Decision 4). No
visitor-facing inference endpoint.

The seam: `ModelRuntime` is an interface, so a future `remote` backend (a quota'd, rate-limited
`POST /models/{id}/run`, opt-in per visitor, off by default) can be added behind it without touching game pages. At
that point it needs its own design (abuse protection, per-visitor quotas, cost caps). Nothing is built for it now.

A related option, compatible with "offline only": **recorded episodes**. The evaluation job already plays every
entrant on held-out seeds. Storing `(seed, actions)` per game costs bytes, not compute, and the Rust game core replays
them deterministically on any device. A visitor whose device can't *run* a model could still *watch a recording
of it*, clearly labelled as a recording. Listed as a late, optional milestone below.

## Tech stack additions (as of September 2026, verify before pinning)

| Piece | Choice | Version | Why |
|---|---|---|---|
| Browser inference | `onnxruntime-web` | 1.30.x | WebGPU EP (recommended path upstream; WebGL/JSEP deprecated) + WASM EP fallback, int4/int8 kernels, external data |
| Export/verify | `onnx`, `onnxruntime` (Python) | match ORT Web's release | Parity checks and evaluation run the same kernel family the browser does |
| Game core | Rust (stable) + PyO3 | PyO3 0.29.x | Python bindings for training/eval |
| Python build | maturin | 1.13.x | PEP 517 backend, so `libs/games` stays a `uv` workspace member |
| WASM build | wasm-bindgen + wasm-pack | latest | Small module, no Emscripten runtime |
| Optional trainer | PyTorch | as needed | Only via `modelpack[torch]`, only for models where scale matters |
| Retired | Pyodide | — | After the Rust port (Decision 2) |

WebGPU availability today: shipping by default in Chrome/Edge (desktop, and Android 12+ on common GPUs), Safari 26
(macOS/iOS/iPadOS), and Firefox on Windows and Apple-silicon macOS. Firefox on Linux/Android is still in progress.
That gap is exactly what Decision 6's reasons exist for.

[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) ·
[ORT WebGPU EP](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) ·
[WebGPU implementation status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) ·
[web.dev: WebGPU in all major browsers](https://web.dev/blog/webgpu-supported-major-browsers) ·
[PyO3](https://github.com/pyo3/pyo3/releases) · [maturin](https://github.com/PyO3/maturin/releases)

## Where things live

- `libs/games/`: becomes the Rust cargo workspace described in Decision 2. The Python import surface is unchanged.
- `libs/modelpack/` (new): manifest schema, exporters (`WeightVector`, `NeatGenome`, `autodiff`, `torch`), parity
  and action-agreement checks, `ModelStore` + `LocalModelStore`.
- `jobs/`: `publish_model.py` (export → verify → publish → catalog). `evaluate.py` evaluates packages and
  variants instead of in-memory objects.
- `apis/backend`: static serving of the local model store in dev, `GET /games/{game}/models` (catalog).
- `apps/frontend`: `app/inference/` (`ModelRuntime`, ORT backends, `native-ts`, `probeDevice`, `matchVariant`,
  shard cache, memory budget), a generalized `session.worker.ts` (game WASM + model runtime), and the wasm-pack
  output of `libs/games/wasm`. COOP/COEP headers in Nuxt's server config.

## Incremental plan

Ordered so each step proves one thing end to end before the next depends on it.

1. **Model packages for today's models.** `libs/modelpack` with the manifest, `WeightVector` + `NeatGenome`
   exporters, max-error + action-agreement checks, `LocalModelStore`, `jobs/publish_model.py`. Publish every current
   Snake leaderboard champion. *Exit:* each package's fp32 variant agrees on 100% of decisions on `snake.score.v1`'s
   200 held-out games (or, where it doesn't, it's entered as its own variant).
2. **ORT in the browser, for those same models.** `ModelRuntime` with `ort-wasm` then `ort-webgpu`, `probeDevice`,
   `matchVariant`, the unsupported-with-reason UI, and the shard cache. Interim: the game still runs in Pyodide
   and observations cross into ORT each tick. That is acceptable at Snake's tick rate, and it exists to prove the
   model pipeline independently of the game port. *Exit:* `/watch` and `/games/snake` play every champion via ORT, and a
   forced "no WebGPU" profile shows the right reasons without breaking the WASM-runnable ones.
3. **Rust Snake.** Core + PyO3 + wasm bindings, PRNG spec, oracle-checked against the Python, `snake.score.v2` and
   re-evaluation, and the worker switches from Pyodide to the Snake WASM for Snake pages.
4. **Everything else in Rust.** Checkers, `reach1d`, `games.baselines`, `explain()` for every observer. Remove Pyodide and
   `py-source`.
5. **The first model that needs this.** A pixel-observation (`L0`, doc 0007) conv policy for Snake, trained via
   `autodiff` once it has convolution or via PyTorch, shipped with fp16 and int8 variants on WebGPU. This is the first real
   exercise of shards, persistence, the size-confirmation click, and the memory budget.
6. **`tinylm` in the browser.** `autodiff` exporter, a byte tokenizer in TS, our own sampling loop, GPU-resident
   KV-cache, int4 variant. This is the scale test toward the ~1 GB ceiling.
7. **Hosting.** Pick R2 or the HF Hub, implement its `ModelStore`, and set COOP/COEP + CORS on the real deploy.
8. *(Optional)* **Recorded episodes** for devices that can't run a model (Decision 7).

## Explicitly out of scope for now

- Visitor-facing server-side inference of any kind (a seam only, Decision 7).
- A from-scratch production WebGPU inference engine (a Learn chapter only).
- WebLLM/MLC, WebNN, and training in the browser.
- Protecting weights from download. Anything that runs client-side is public, which is fine for this project.
- Cross-browser timing comparisons on leaderboards. Browser µs/decision varies by GPU, driver, and tab state, so
  leaderboards keep server-side, hardware-fingerprinted timings plus FLOPs (doc 0007's "clocks second").

## Open questions

- **Host:** R2 + CDN vs. the Hugging Face Hub (Decision 5). This can be decided at deploy time; step 7 depends on it.
- **Download threshold** for the explicit-confirm prompt (50 MB is a starting guess), and whether a
  "Wi-Fi only"-style preference is worth adding.
- **Does anything still need Python in the browser after step 4?** A future "edit the observer and watch it retrain"
  Learn chapter would. If it ever comes, Pyodide returns as an opt-in chapter dependency, not the game path.

## Implementation notes (steps 1–6, September 2026)

What was built follows the decisions above. Where it deviated, and why:

- **An fp64 variant for evolved networks** (not in the plan). The evolved policies are float64
  throughout, and an fp32 export changed moves for five of ten Snake champions (one old grid champion
  on 11% of moves) despite ~1e-6 numeric error. fp64 runs on the WASM backend (WGSL has no f64) and
  agrees on every move for all but one champion, whose saturated `tanh` outputs tie *exactly*, so a
  one-ulp difference in `tanh` breaks ties differently (99.5%). Rule applied: the leaderboard entrant
  plays through a variant that agrees on every decision *and* every score; any other variant is its
  own entrant (`run:<id>@fp32`), and a champion with no exact variant stays ranked through the Python
  forward pass.
- **Evaluation runs the package** (as decided), via ONNX Runtime CPU: the same scores for exact
  variants, plus rows for inexact ones.
- **On-demand export for unpublished champions** (`POST /runs/{id}/artifacts/{ref}/package`). Live
  training and pinned generations still needed a way to play once Pyodide was gone; exporting on
  demand keeps one runtime deciding every move instead of reviving a TypeScript forward pass. It's
  content-addressed, so asking twice costs nothing. It's development-time only and never a
  leaderboard entrant.
- **Pyodide retired in step 3, not step 4**: nothing else in the browser needed it once Snake ran on
  the WASM core. Step 4 then moved Checkers and Reach1D anyway, with WASM bindings for a future
  Checkers page. `explain()` wasn't ported because it doesn't exist yet (doc 0007 plan step 3).
- **Checkers' move order had to be ported, not just its rules.** Strategies index into
  `legal_moves()`, which the Python generated in dict insertion order; the Rust board is an
  insertion-ordered map with dict semantics. Parity: 70 full games; `checkers_round_robin.py`
  reproduces the Learn chapter's published numbers exactly.
- **The TinyLM exporter is per layer, not a trace of autodiff's op graph.** A trace fixes (batch,
  seq_len) and resolves embedding lookups to concrete rows, so it can't take a variable-length prompt
  or carry a KV cache. One graph serves prefill and per-token decoding, with the causal mask built
  in-graph from the cache length.
- **Step 5 was done as infrastructure only.** A pixel-observation conv policy needs convolution in
  `libs/autodiff` (or PyTorch) and is research of its own. The step's stated purpose, exercising
  shards, the confirmation click, persistence and WebGPU at size, is covered by `jobs/scale_test_package.py`:
  an 85M-parameter randomly initialized TinyLM (341 MB fp32 in 12 shards, 86 MB int8), clearly
  labelled as noise, on `/dev/inference`.
- **Found only in a browser:** ONNX Runtime Web resolves `Reshape` shapes before attaching external
  data, so int32/int64 and small tensors stay inline (the first rule also kept int8 *weights* inline;
  the scale test exposed that). Other bugs found by running it: a Vue proxy can't be `postMessage`d,
  and a late verdict of "unsupported" has to stop whatever already started playing.

### Measured (this machine: Windows, Chrome, NVIDIA Blackwell GPU; dev server, localhost)

| What | Result |
|---|---|
| Game core in the browser | 47 KB WASM (all three games), vs. Pyodide's multi-MB, ~11 s cold start |
| Game core in Python | ~5x faster per Snake step than the pure-Python original (0.79 vs 4.14 µs, greedy policy included) |
| ORT runtime download | WASM build ≈3.7 MB gzipped, WebGPU ≈6.6 MB, only the one needed, cached after |
| Snake champion packages | 2–21 KB; load + self-test ~170–300 ms; fp64 self-test error ~1e-16 |
| TinyLM (79K params) | fp32 327 KB, 100% top-1 agreement; int8 131 KB, 98.2%. ~2,000–2,600 chars/s on WASM, 133 on WebGPU |
| Scale test (85M params) | fp32 341 MB on WebGPU and int8 86 MB on single-threaded WASM both ~83 chars/s; load 2.4 s cold, 1.4 s cached |

Takeaways: small packages belong on WASM (GPU dispatch dominates at batch 1); int8 on WASM is a
competitive default for mid-sized LMs at a quarter of the download; and per-decision agreement, not
numeric error, is the number that says whether an export is faithful.

### Not done yet

- **Step 7, hosting**: choose R2 or the Hugging Face Hub and implement its `ModelStore` (the layout,
  `base_url` and `REDQUEEN_MODELS_BASE_URL` seam exist). Serve the app with COOP/COEP so multithreaded
  WASM turns on (`/dev/inference` shows "threads: no" today), and make sure every cross-origin asset
  (fonts included) survives `require-corp` or `credentialless`.
- **Step 8, recorded episodes**, for devices that can't run a model.
- **Pixel-observation (L0) conv policy** (step 5's model), an **fp16 WebGPU variant** (the scale-test
  GPU reports `shader-f16`), and a **memory budget across several loaded models**. Every page loads
  one model at a time today, so the budget isn't needed yet.
- `LocalModelStore.collect_garbage()` exists (republishing seals new packages); a remote store will
  need the same.
