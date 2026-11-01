# 0023 — Research log and process traces: watching experiments and algorithms as they run

Status: **Phase R built** (2026-11-01); the traces are proposed. Copy Cat (0022) is the first consumer, but nothing in it is
specific to Copy Cat except the first algorithms it serves.
Relates to: [0005](0005-frontend-and-api-contracts.md) (one app, one API), [0007](0007-representations-leaderboards-and-tradeoffs.md)
(an entry as a vector of measurements), [0009](0009-client-side-inference-at-scale.md) (WASM and ONNX in the browser),
[0018](0018-workloads-api-and-cli.md) (jobs), [0019](0019-static-hosting.md) (the static copy),
[0022](0022-copy-cat.md) (the ladder and the staged plan).

## Context

Two things are missing, and they are different sizes.

**Nobody can see the research except through chat.** Every change to Copy Cat is measured by `copycat.report`, and each
report is a JSON file under `<data dir>/experiments/` with a commit, a note and per-image scores. Directing the work means
reading those numbers as they are relayed. There is no view of how the score has moved over the changes, which images a
change moved, or how far each stage's target is.

**Nobody can see an algorithm work.** The page shows finished answers and one snapshot of the structured pipeline. Stage by
stage, `stages()` computes far more than it keeps. The 0022 fix "An outline as seen" turned on a part costing 1,605 bits
completed and 381 as seen, and only temporary logging showed it. The search and gradient rungs (0022's ladder, steps 3-4)
would be much harder to debug this way, and the families below refine in very different ways.

## Decision 1: the research log comes first, and it is mostly reading what already exists

A page for `copycat.report`'s reports, before any new recording.

- **`GET /games/copycat/reports`** lists reports, from the files `copycat.report` already writes: name, commit, note, and
  each group's mean, worst, columns and by-severity row.
- **`GET /games/copycat/reports/{name}`** adds per-image scores and the comparison with its baseline (`compare`, which the
  protocol already has, moved to where the route can call it).
- **The page** (`/research/copycat`, unlinked at first):
  - progress: each group's mean over the reports, in order, with the staged plan's targets drawn as lines;
  - one change: its note, its table (before, after, images up and down), and its movers, sorted worst first, each opening
    the image with input, ours and clean;
  - a report is marked kept, or measured and not kept. Reports don't say today: a `decision` field joins the spec's params,
    and old reports read as unmarked.
- **The page only reads at first.** Submitting a report goes through the job API that already exists (`POST /jobs`), when
  the scheduler is running. So "try this next" can become a button later without a new mechanism.
- **Local first.** Reports aren't part of the public site until a person decides they should be, because publishing is
  outward-facing and reports name work that was rejected. The routes follow the static export's rules (path-only, no
  query strings), so adding them is one loop in `static_site.py` when that's wanted.

Held-out scores appear only where a report already has them, and the page must show when a report looked at held-out, so
"once per change" stays visible.

**As built**, three things differ from the above:

- **A decision is made after the measurement,** so a spec's `decision` isn't enough: `PATCH /games/copycat/reports/{name}`
  writes it into the report, and the page has the control. That is the page's one write.
- **Reports keep their SVGs** (`<out>.svgs.json`, about 0.7 MB beside the report), since "ours" has to be what that change
  drew, not what today's build draws. Older reports show the image and its clean SVG, and say theirs wasn't kept.
- **Held-out images aren't served one by one,** only their means: `GET .../reports/{name}/images/{set}/{split}/{seed}` is a
  404 for them, so the page can't become a way to debug on them.

## Decision 2: one trace format and one viewer, with no per-algorithm UI

An algorithm records typed state on named tracks, against shared timelines. The viewer lays out, scrubs and compares those
tracks without knowing what produced them. This follows Rerun's data model (typed data at paths, several timelines,
latest-at queries, a layout kept apart from the data, a table for anything unknown). We don't embed Rerun itself: its
viewer is a canvas that can't share our components, and its file format is still changing.

**The unit that matters is a per-element ledger with its rejected alternatives.** Most experiment trackers show media and
loss curves. Those would not have caught 1,605 against 381. The same structure underlies Optimize & Reduce's removal ranks and
every search rung.

Ten closed track kinds, each with one Vue renderer written once:

| Kind | Meaning |
|---|---|
| `vector` | The current document as a keyed element map; each element may be marked observed, completed or invented |
| `raster` | Input, residual, or an inpainted image |
| `labelmap` | An index image and its palette |
| `overlay` | Control points, arrows, proposals on a named canvas |
| `scalar` | A series with a unit, a better direction and an axis |
| `ledger` | Per-element named numbers: bits, lost if removed, visibility |
| `alternatives` | Competing states at one moment with scores and a winner; with descriptors, an archive |
| `graph` | Nodes and edges over time: an occlusion graph, a search tree |
| `text` | Notes, a growing SVG stream |
| `table` | Anything else, never an error |

Beside them, **spans** (the stage tree and restarts) and **events** (a proposal accepted, with its delta).

Rules that keep it general:
- A kind is added only when a second algorithm needs it.
- An algorithm's name never appears in viewer code.
- Explanations go in the explainer registry: a `stage` concept joins `ExplainKind`, and a span refers to one.

## Decision 3: an algorithm can adopt the hook partway, and the page says how far

| Level | The author supplies | Examples | What the viewer shows |
|---|---|---|---|
| 0: outcome | Input and final SVG | VTracer today | Before and after, labelled "no process recorded" |
| 1: snapshots | SVGs, optional scalars | primitive's frames | A scrubber and derived scores |
| 2: structure | Spans, per-stage state | A recorded external run | A timeline strip and channels |
| 3: decisions | Ledger, alternatives, events | The structured pipeline, search rungs | Full diagnosis |

Re-running VTracer at several settings and animating the results would show something it doesn't do. So level 0 is shown as
what it is.

## Decision 4: frames are scored by one evaluator, not by the algorithm

Any trace with a `vector` track gets clean fidelity, alpha IoU, node counts and L from one Rust code path, so every family
shares its y-axes whatever its own loss is.
- **Comparison runs on budgets, not steps:** nodes, bits, time. A mutation and a gradient step aren't comparable.
- **Fit to input and fit to clean are drawn together** where a clean answer exists. A loss still falling while clean fidelity
  turns down would be noise being fitted. That's an untested idea: no paper we read tests a vector optimizer on noisy input.
- **The final frame must equal the scored SVG**, byte for byte. The page's existing badge extends to the whole film.

## Decision 5: the recorder is invisible when off

A generic `Recorder` with a no-op default in the Rust core.
- It must not draw PCG32 or alter a comparison. `jobs/tests/test_golden_eval.py` holds it to that.
- It stays off the timed path (`image_us` wraps `trace()`), so stored traces come from a second, untimed pass.
- By default it records accepted state, since coordinate descent evaluates far more than it keeps.
- Libraries stay free of telemetry: job glue writes the artifacts. Non-Rust entrants use a thin Python writer of the same
  schema.

## Decision 6: parameters come from a schema, and each declares what changing it costs

A pydantic `TraceParams` per algorithm is the source of truth, pinned to the Rust defaults by a test, with its schema
generated and checked in the way `api.gen.ts` is. A small form renderer over `UiRange`, `UiSelect` and `UiCheck` draws it.

| Tier | What changes | Works on the static site |
|---|---|---|
| View | Nothing is recomputed (channel, overlay, axis) | Always |
| Replay | Picks among recorded values | For recorded values |
| Re-run | Runs again in a worker, from the first span the change invalidates | WASM entrants only |

Any change from the scored settings shows "not what was scored". A slider on a held-out image is a teaching tool, labelled
so, because a pleasant parameter explorer is how tuning leaks into held-out data. What it suggests is re-tested with
`copycat.report` on `train`.

## Decision 7: traces get their own route

`GET /games/copycat/targets/{seed}/traces/{entrant}`, over the artifact store, with content-addressed chunks copied to R2
by a new loop in `static_site.py`. Not `?kind=trace` (static mode strips query strings, and baselines have no run), and not
in the evaluation record (`/targets/{seed}` is already megabytes). Chunks are keyframes and patches over keyed elements, as
rrweb does, not diffs of SVG text.

## The algorithm families this has to carry

| Family | How it refines | In a browser | What you'd watch |
|---|---|---|---|
| Classical tracers | Stages | Yes | Palette, contours, curves |
| Primitive and greedy search | Adds shapes | Yes | A canvas growing |
| Structured MDL (0022, built) | Stages | Yes | Bit costs, candidates per part, exploded stack |
| Structure search (0022 step 3) | Moves on structure | Yes, with incremental scoring | Moves accepted and rejected, L falling |
| Gradient refinement (0022 step 4) | Every point moves | Plausible | Edges sliding, error fading |
| Layered and amodal peeling | Removes a part, invents what's beneath | Segmentation only | A filmstrip of parts and underlays |
| Models that write SVG | Adds tokens or paths | No | SVG typed out |
| RL painters and program synthesis | A learned sequence | Inference only | Strokes, program text |

In the order we would try them (a judgement, to be settled by `copycat.report`):
1. structure search under L, with Optimize & Reduce's "lost if removed" as one move;
2. opaque, palette-locked refinement on an exact-coverage renderer (Windfoil-style, hand-written in Rust);
3. MAP-Elites over node count and error, which we found no prior use of on SVG;
4. the non-neural parts of layered papers: SemLayer's ILP ordering and curve reuse, LayerD's palette snapping;
5. an optional fine-tuned matting proposal through ONNX;
6. recorded LayerPeeler or SemLayer runs, as labelled exhibits.

## Phases

Each has a gate, so a wrong bet shows early.

| Phase | Build | Gate |
|---|---|---|
| R: research log | The two report routes, the page, `decision` in reports | Every report since `copycat-ae2f606` is readable, with no hand-kept data |
| 0: measure | A minimal recorder on the structured pipeline and the hill climb | Bytes and frames per image are known before the schema is frozen |
| 1: one consumer | Schema, `Recorder`, WASM entry, viewer, live only | The viewer reproduces 1,605 against 381 with no logging; goldens unchanged; final frame equals the scored SVG |
| 2: second consumer | Hill-climb binding and trace; a shared worker shell | The hill climb needs no new `kind` |
| 3: stored and compared | The route, the static loop, level 0 for VTracer, three comparison levels | The static site works with `?device=nowasm` |
| 4: parameters | `TraceParams`, the form, the tiers, staged re-runs | The badge fires on any change |
| 5: search | 0022's step 3 | **No new Vue components beyond a shared kind** |
| 6: gradients | 0022's step 4 | Fit to input and to clean, on noisy images, through `copycat.report` |
| 7: exhibits | A recorded external method | Licences and terms read by a person; labelled as recorded |

## Open questions

- **Trace size.** Unmeasured. "A few hundred kilobytes per image" and "a keyframe every 50 frames" are guesses; phase 0 replaces them.
- **Search and gradient needs.** The schema's fit to rungs 3 and 4 is a prediction until they exist.
- **Publishing the research log.** Whether rejected work belongs on the public site is a person's decision, later.

## Sources

The figures in this doc's reasoning came through a page-summarising tool. Check any against its source before public use.

- Rerun's data model: <https://rerun.io/docs/concepts/entity-component>
- rrweb: <https://github.com/rrweb-io/rrweb>
- Optimize & Reduce: <https://arxiv.org/abs/2312.11334>
- Windfoil: <https://arxiv.org/abs/2610.02468>
- CubicSplat: <https://arxiv.org/abs/2608.20803>
- SemLayer: <https://arxiv.org/abs/2603.24039>
- LayerD: <https://arxiv.org/abs/2509.25134>
- LayerPeeler: <https://arxiv.org/abs/2505.23740>
- AmodalSVG: <https://arxiv.org/abs/2604.10940>
