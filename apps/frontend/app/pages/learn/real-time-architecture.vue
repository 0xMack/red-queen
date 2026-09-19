<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
const callbackCode = `# jobs/snake_neuro_run.py -- the only place evolve and telemetry meet
def on_generation(summary: GenerationSummary) -> None:
    champion_ref = f"{run_id}-gen{summary.generation}"
    artifacts.put_program(champion_ref, summary.champion.to_json().encode("utf-8"))
    metrics.record_generation(GenerationStats(
        run_id=run_id, generation=summary.generation, timestamp=time.time(),
        best_fitness=summary.best_fitness, mean_fitness=summary.mean_fitness,
        worst_fitness=summary.worst_fitness, diversity=summary.diversity,
        champion_ref=champion_ref, held_out_score=held_out_score,
    ))

evolve(population, ..., on_generation=[on_generation, cost_meter, control_callback])`

const subscribeCode = `class FileMetricsStore:
    def record_generation(self, stats):            # the job: append one JSON line
        with self._path(stats.run_id).open("a") as f:
            f.write(stats.model_dump_json() + "\\n")

    def subscribe(self, run_id, since_generation=0):   # the API: a live tail, never returns
        last_generation = since_generation - 1
        while True:
            for stats in self._read_all(self._path(run_id)):
                if stats.generation > last_generation:
                    last_generation = stats.generation
                    yield stats
            time.sleep(self._poll_interval)         # 0.5s`

const sseCode = `async def _stream_generation_stats(metrics, run_id, since_generation):
    # subscribe() is synchronous and sleeps between polls: run each next() on a worker thread
    # so it never blocks the event loop, and abandon it the moment the browser disconnects.
    iterator = metrics.subscribe(run_id, since_generation=since_generation)
    while True:
        stats = await anyio.to_thread.run_sync(next, iterator, abandon_on_cancel=True)
        yield {"event": "generation", "data": stats.model_dump_json()}`

const browserCode = `// app/stores/metricsStream.ts -- backfill, then live, with no gap and no duplicates
history.value = await $fetch(\`/runs/\${id}/metrics/history\`)
const sinceGeneration = (history.value.at(-1)?.generation ?? -1) + 1
source = new EventSource(\`\${apiBase}/runs/\${id}/metrics/stream?since_generation=\${sinceGeneration}\`)
source.addEventListener("generation", (event) => history.value.push(JSON.parse(event.data)))`

const controlCode = `def make_control_callback(registry, run_id, poll_interval=0.2):
    def on_generation(_summary):
        while registry.get_run(run_id).status == "paused":
            time.sleep(poll_interval)
    return on_generation`
</script>

<template>
  <article class="prose-chapter">
    <p>
      Every run on this site streams its progress while it trains: the fitness chart grows a point per
      generation, the champion on the board swaps to the newest one, and you can pause a run from the
      page. None of that needs a message queue, a websocket server, or a database server. This chapter
      is how it actually works -- and why each piece is as boring as possible.
    </p>

    <h2>The job never knows about the web</h2>
    <p>
      A training run is an ordinary Python process calling <code>evolve()</code>. The evolution loop
      has <em>zero</em> dependency on telemetry: it just calls whatever callbacks it was given after each
      generation, with a summary that holds the real champion genome. The job script is the only code
      that knows about both sides, and it's a few lines of glue:
    </p>
    <CodeBlock lang="python" :code="callbackCode" />
    <p>
      That split is what lets <code>evolve</code> be unit-tested with no files or databases at all, run
      unchanged in a notebook, and -- as the Snake pages show -- run in the browser.
    </p>

    <h2>Three boring stores</h2>
    <p>
      Telemetry is three small interfaces, each with a deliberately simple local implementation:
    </p>
    <ul>
      <li>
        <strong>Run registry</strong> -- one SQLite row per run: config, status, timestamps, a summary.
        SQLite's WAL mode lets one writer (the job) and many readers (the API) share it safely.
      </li>
      <li>
        <strong>Metrics</strong> -- one JSON-lines file per run, one line per generation, append-only.
      </li>
      <li>
        <strong>Artifacts</strong> -- one file per stored champion, opaque bytes. The API doesn't know or
        care whether it's a linear program's text or a network's weights.
      </li>
    </ul>
    <p>
      Each is a Python <code>Protocol</code>, so a hosted backend (Redis, a real database) could replace
      any of them without touching the job, the API routes, or the browser (docs/design/0002). None has
      been needed yet.
    </p>

    <h2>Backfill, then live</h2>
    <p>
      A page opened halfway through a run needs everything so far <em>and</em> everything from now on,
      with no gap and nothing twice. The browser asks for the history first, then opens a
      <strong>Server-Sent Events</strong> stream starting at the next generation:
    </p>
    <CodeBlock lang="typescript" :code="browserCode" />
    <p>
      On the server, "the stream" is just the metrics file being tailed. <code>subscribe()</code> polls it
      every half second -- no OS file-watching APIs, so it works the same on every platform, and at one
      write per generation polling costs nothing:
    </p>
    <CodeBlock lang="python" :code="subscribeCode" />
    <p>
      That generator blocks while it sleeps, and the API server is async, so each <code>next()</code> runs
      on a worker thread -- and is abandoned the moment the browser goes away:
    </p>
    <CodeBlock lang="python" :code="sseCode" />
    <p>This is that stream, connected for real:</p>
    <ClientOnly><LiveEventLog /></ClientOnly>
    <Callout variant="note" title="SSE, not websockets">
      Data only flows one way (job → browser), so Server-Sent Events are enough: plain HTTP, built-in
      reconnection in the browser's <code>EventSource</code>, and nothing to configure. The one case that
      flows the other way -- pause/resume -- is an ordinary POST.
    </Callout>

    <h2>Pausing a run without new plumbing</h2>
    <p>
      The API and the training job are separate processes. The obvious way to let one pause the other is
      a new channel between them. Instead, both already share the run registry, and it already has a
      <code>status</code> column with a <code>"paused"</code> value. The API flips it; the job checks it
      after each generation:
    </p>
    <CodeBlock lang="python" :code="controlCode" />
    <p>
      "Step one generation" doesn't even need a third state: the API resumes the run, waits until exactly
      one new generation appears in the metrics file, and pauses it again. The job only ever understands
      two states.
    </p>

    <h2>Replaying champions in the browser</h2>
    <p>
      Watching a champion play doesn't call the server per move -- a visitor's own device does the
      inference, so watching costs the server nothing per frame. Two pieces run inside a Web Worker. The
      <strong>game</strong> is <code>libs/games</code>' Rust core compiled to WebAssembly (31 KB): the same
      code training runs through Python bindings, so the snake you watch plays exactly the game the
      leaderboard scored. The <strong>model</strong> is a content-addressed package -- an ONNX graph plus
      its weights, verified by hash and cached forever -- run by ONNX Runtime Web on WebAssembly or the GPU,
      whichever this device supports. If it supports neither, the page says so and why, rather than
      failing (docs/design/0009).
    </p>
    <Callout variant="finding" title="Precision changed the moves">
      Exporting the evolved networks in float32 -- what a GPU runs -- made five of ten champions choose
      a different move somewhere in their 200 held-out games: the float64 networks' argmax sat on near-ties
      that rounding flipped (one old grid champion disagreed on 11% of its moves). Every package now
      records how often each variant agrees with the trained model, the leaderboard entrant plays
      through a variant that agrees on every move, and a variant that doesn't is ranked separately.
    </Callout>
    <Callout variant="finding" title="Streaming made a real problem visible">
      The live curve of training fitness against a held-out game score (recorded every 10 generations)
      showed the Snake champion's real score peaking around generation 20 and then <em>falling</em> while
      its training fitness kept rising: it was memorizing its five training games. Nothing about the
      final number revealed that; watching the two curves diverge did. That's the case for building the
      visibility first (docs/design/0007).
    </Callout>
  </article>
</template>
