<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
// The numbers are data/copycatVersions.ts's: the MDL tracer's versions, each held-out score taken once per change.
import { COPYCAT_VERSIONS, versionsThatMoved } from "~/data/copycatVersions"

const SYNTHETIC = versionsThatMoved("synthetic")
const ICONS = versionsThatMoved("icons")
const seriesOf = (key: string, label: string, color: string, points: { score: number }[]) => [
  { key, label, color, values: points.map((p) => p.score), width: 2.5 },
]
const syntheticSeries = seriesOf("synthetic", "held-out synthetic", palette.queen400, SYNTHETIC)
const iconSeries = seriesOf("icons", "held-out icons", palette.signal400, ICONS)
const versions = (points: { version: number }[]) => points.map((p) => p.version)
const score = (v: number | undefined) => (v === undefined ? "--" : v.toFixed(2))
</script>

<template>
  <article class="prose-chapter">
    <p>
      <NuxtLink to="/learn/tracing-images">The tracing pipeline</NuxtLink> has no training run and no neural network. It is a fixed
      program with tuned constants, and every piece of it costs something: code, tracing time, and a new way to be wrong. So
      each piece had to show it was worth that. This is how those decisions went, including a few that came out "no".
    </p>

    <UiFigure title="Fidelity (dB) on images never tuned on, change by change" kind="Results">
      <div class="grid gap-6 md:grid-cols-2">
        <div>
          <UiLegend :series="syntheticSeries" />
          <LineChart class="mt-2" x-label="version" :x="versions(SYNTHETIC)" :series="syntheticSeries" :height="200" :format="(v: number) => v.toFixed(1)" />
        </div>
        <div>
          <UiLegend :series="iconSeries" />
          <LineChart class="mt-2" x-label="version" :x="versions(ICONS)" :series="iconSeries" :height="200" :format="(v: number) => v.toFixed(1)" />
        </div>
      </div>
      <template #caption>
        Each point is a version that moved that set's held-out score, scored once when its change was done; the tracer's
        number goes up by one with every change kept, and every version is listed at the end of this chapter.
        For comparison, VTracer scores 20.18 dB on held-out <code>synthetic</code>.
      </template>
    </UiFigure>

    <h2>Find the error before adding anything</h2>
    <p>
      The tempting move is to add whatever seems missing. The first results said why not to. <code>heavy</code> images scored
      under 23 dB, and we blamed colour drift. Then we measured it with an oracle on the training images: give every output
      shape the true colour from the clean SVG, and see what that buys.
    </p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-4 py-2 font-medium">Corruption</th>
            <th class="px-4 py-2 text-right font-medium">Traced</th>
            <th class="px-4 py-2 text-right font-medium">With true colours</th>
            <th class="px-4 py-2 text-right font-medium">A perfect copy of the warped image</th>
          </tr>
        </thead>
        <tbody class="num">
          <tr class="border-b border-line/60"><td class="px-4 py-2 font-sans">clean</td><td class="px-4 py-2 text-right">35.3</td><td class="px-4 py-2 text-right">35.3</td><td class="px-4 py-2 text-right">60</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2 font-sans">light</td><td class="px-4 py-2 text-right">31.6</td><td class="px-4 py-2 text-right">31.9</td><td class="px-4 py-2 text-right">42.1</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2 font-sans">medium</td><td class="px-4 py-2 text-right">26.8</td><td class="px-4 py-2 text-right">27.2</td><td class="px-4 py-2 text-right">30.1</td></tr>
          <tr><td class="px-4 py-2 font-sans">heavy</td><td class="px-4 py-2 text-right">22.9</td><td class="px-4 py-2 text-right">23.6</td><td class="px-4 py-2 text-right">24.4</td></tr>
        </tbody>
      </table>
    </div>
    <Callout variant="finding" title="Colour was never the problem">
      True colours add 0.3 dB overall. On <code>heavy</code>, even a pixel-perfect copy of the input's geometry would only reach
      24.4 dB, because the warp moved the edges. Fitting the input harder can't pass that. On <code>clean</code> and
      <code>light</code> there was plenty of room, so that is where the work went.
    </Callout>
    <p>
      Almost every later change started the same way: an oracle or a per-pixel split of the error, naming what a perfect answer
      would gain. A change then had to beat the version before it on the training images, and got one score on held-out images
      at the end.
    </p>

    <h2>Each rung against the last</h2>
    <p>On <code>synthetic</code>, the held-out score and the time it cost:</p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-4 py-2 font-medium">What the evidence said</th>
            <th class="px-4 py-2 font-medium">What was added</th>
            <th class="px-4 py-2 text-right font-medium">Held-out</th>
            <th class="px-4 py-2 text-right font-medium">Per image</th>
          </tr>
        </thead>
        <tbody>
          <tr class="border-b border-line/60"><td class="px-4 py-2">On clean images, all the error is at boundaries</td><td class="px-4 py-2">Nudge every number of the likely winners while the description shrinks</td><td class="num px-4 py-2 text-right">29.0 → 30.8</td><td class="num px-4 py-2 text-right">76 → 152 ms</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Exact geometry for smooth blobs alone would be worth +7.5 dB on clean; 24 of 46 had come back as polygons</td><td class="px-4 py-2">A spline described by its knots</td><td class="num px-4 py-2 text-right">30.8 → 31.38</td><td class="num px-4 py-2 text-right">219 ms</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">20 output shapes were the best match for two clean parts</td><td class="px-4 py-2">Cut a part between two notches in its outline</td><td class="num px-4 py-2 text-right">31.38 → 31.43</td><td class="num px-4 py-2 text-right">223 ms</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Blobs still worth +5.7 dB, and the family could draw them: the search was what fell short</td><td class="px-4 py-2">Solve for the knots by least squares</td><td class="num px-4 py-2 text-right">31.43 → 31.99</td><td class="num px-4 py-2 text-right">293 ms</td></tr>
          <tr><td class="px-4 py-2">On noisy images, outlines were following the noise (<code>heavy</code> turning 1.25× the clean SVG's)</td><td class="px-4 py-2">Keep a point only if it pays for a move's worth of bits</td><td class="num px-4 py-2 text-right">31.99 → 32.00, turning 1.18×</td><td class="num px-4 py-2 text-right">--</td></tr>
        </tbody>
      </table>
    </div>
    <p>
      Inside each rung, the same test applied to each option. Six spline knot counts beat four by 0.07 dB for 10% more time, so
      four stayed. On the training images, refining every outline reached 30.31 dB at 742 ms per image, while refining only the
      two shortest, before choosing, reached 30.87 at 99 ms: better and cheaper.
    </p>

    <h2>The one that didn't carry over</h2>
    <p>
      <code>synthetic</code>'s polygons are regular, and they were recovering worst. So a regular-polygon candidate looked
      obvious: n sides, a centre, a radius and a turn. It worked on the training images, with polygon overlap on
      <code>medium</code> rising from 0.73 to 0.80 and fidelity from 31.92 to 32.14.
    </p>
    <Callout variant="warning" title="Train +0.22 dB, held-out -0.06 dB">
      Scored once on held-out images, it fell from 31.38 to 31.32 dB, so the code was removed. Looking at why on the training
      images was worth more than the candidate. Two of the worst polygons were so hidden that too little outline showed to pin a
      regular shape. Another was a visible hexagon merged with a neighbour of its colour, which no shape prior can fix. That
      case is what the cut in the table above came from.
    </Callout>

    <h2>A new set, and the same pipeline at 18 dB</h2>
    <p>
      <code>synthetic</code> is ours, and tuning to it has limits. So we added <code>icons</code>, real Twemoji. The pipeline
      scored 18 dB against 32 on <code>synthetic</code>. Almost none of the fixes were about shapes:
    </p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-4 py-2 font-medium">What was going wrong</th>
            <th class="px-4 py-2 font-medium">What changed</th>
            <th class="px-4 py-2 text-right font-medium">Held-out icons</th>
          </tr>
        </thead>
        <tbody>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Two eyes joined across a face made one shape, painted under the face: both vanished</td><td class="px-4 py-2">A piece with a part beneath it is never joined</td><td class="num px-4 py-2 text-right">18.52 → 18.77</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">A tile covering the border was taken for the background and deleted</td><td class="px-4 py-2">The background must reach the corners</td><td class="num px-4 py-2 text-right">18.77 → 19.41</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">White arrows came out under their tile, because a raw pixel count favours whichever part is larger</td><td class="px-4 py-2">Compare the fraction of each part covered</td><td class="num px-4 py-2 text-right">19.41 → 20.88</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Noise or a speck on one corner broke the background fit, costing whole tiles</td><td class="px-4 py-2">Fit from the three corners that agree</td><td class="num px-4 py-2 text-right">20.88 → 21.37</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Pale parts close to a near-white background merged into it</td><td class="px-4 py-2">Give a light region back if it stands out and touches the subject</td><td class="num px-4 py-2 text-right">21.37 → 22.63</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">A ring came back as a disc</td><td class="px-4 py-2">Cut holes, while the whole stays shorter</td><td class="num px-4 py-2 text-right">22.63 → 23.32</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Keys came back as circles, and a mouth lost its tips</td><td class="px-4 py-2">Rounded rectangles and tight splines, each required to fit and not just be short</td><td class="num px-4 py-2 text-right">23.32 → 23.51</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">A rail lost its far side, and nothing noticed</td><td class="px-4 py-2">An uncovered piece gets its own shape if that is shorter</td><td class="num px-4 py-2 text-right">23.51 → 23.60</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2">Close colours an artist meant to differ were forced together</td><td class="px-4 py-2">Keep two flat colours apart, below a noise gate</td><td class="num px-4 py-2 text-right">23.60 → 23.60</td></tr>
          <tr><td class="px-4 py-2">A thin curved part cost 1,605 bits when completed, 381 as seen</td><td class="px-4 py-2">Try the outline as seen for a final shape, if far shorter</td><td class="num px-4 py-2 text-right">23.60 → 23.68</td></tr>
        </tbody>
      </table>
    </div>
    <p>
      Several of these are repairs, and they say so in the record. "Lost pieces" and "an outline as seen" treat symptoms of a
      paint order that is still wrong: the score moved, but layer recovery barely did. A number going up isn't the same as the
      design being recovered, which is why layer recovery is measured apart.
    </p>

    <h2>When a number can't be trusted</h2>
    <p>
      The training images are few enough to overfit, and the rule for held-out images is that you never debug on them. The
      "two flat colours" rule ran into both. It looked clean on the training icons, with 10 images up and none down. Held-out
      <code>synthetic</code> then fell from 32.469 to 32.413.
    </p>
    <p>
      Rather than open those held-out images, we scored 400 fresh seeds. That showed 11 images moving: 6 up and 5 down, with the
      5 all at <code>medium</code> or <code>heavy</code>. The rule only applies below a noise level now, and those 400 seeds are a
      permanent validation set that is free to debug on.
    </p>
    <Callout variant="note" title="A held-out score can also say nothing">
      One change affected 7 of 128 training icons and left all 24 held-out icons identical, to every digit. That isn't
      confirmation, just a set too small to see the change. Even the flat-colours gate is weaker evidence than a first look,
      since it was added after one. The plan includes a bigger icon set for this reason.
    </Callout>

    <h2>Two dead ends worth knowing</h2>
    <p>Most rejected attempts were just the wrong setting of something. These two taught something, and both were implemented correctly:</p>
    <ul>
      <li>
        <strong>Searching the paint order.</strong> Flip two touching parts whenever the whole description gets shorter. With a
        margin it worked, reaching the same gain as the much simpler "outline as seen" rule, at twice the tracing time. So search
        didn't earn its place here, and the reason it had seemed necessary, a misdiagnosis of the tram image, fell apart along the way.
      </li>
      <li>
        <strong>Loosening what may be joined.</strong> It won about 1 dB on four images and lost 3.6 dB on one, where a red slash on
        a tail joined the tail, and 2.7 dB on another, where two soles joined into a sliver. Neither of the two signals tried, how much of the surroundings the covering part owns and how many
        bits the join saves, tells the good joins from the bad. A single fit's bits can't separate a real split from a real detail.
      </li>
    </ul>

    <h2>Measure a rung before building it</h2>
    <p>
      The plan's next rung was a search over the whole structure: delete a shape, swap two in paint order, recolour one. Before
      building it, we ran those moves judged by the answer itself, the clean SVG, which no real search could see. Even that
      oracle won under 0.2 dB on every set, so the search was never built.
    </p>
    <p>
      Setting each kind of pixel right, one kind at a time, showed where the error was: edges and colours, not structure. Edges
      were off by a fraction of a pixel, with no bias either way, because each part had been fitted to its own mask, alone. So
      the rung became a polish: every shape nudged against the whole image as drawn, in the numbers it was fitted in.
    </p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-4 py-2 font-medium">Held-out, scored once</th>
            <th class="px-4 py-2 text-right font-medium">Before</th>
            <th class="px-4 py-2 text-right font-medium">After polish</th>
          </tr>
        </thead>
        <tbody>
          <tr class="border-b border-line/60"><td class="px-4 py-2"><code>synthetic</code></td><td class="num px-4 py-2 text-right">32.47</td><td class="num px-4 py-2 text-right">34.67</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2"><code>icons</code></td><td class="num px-4 py-2 text-right">24.49</td><td class="num px-4 py-2 text-right">25.09</td></tr>
          <tr><td class="px-4 py-2"><code>broad</code></td><td class="num px-4 py-2 text-right">26.92</td><td class="num px-4 py-2 text-right">27.95</td></tr>
        </tbody>
      </table>
    </div>
    <Callout variant="note" title="Polishing the right numbers">
      Polished as its raw cubics, a spline gained as much fidelity but lost its smoothness, since a smooth knot could turn into a
      corner (turning 1.07× to 1.21× the clean SVG's). Polished by its knots, it stays a spline. Polishing the colours too lost
      0.13 dB: fitting a colour to the image fits the corruption's colour drift, which the clean SVG doesn't have.
    </Callout>

    <h2>A point where the outline needs one</h2>
    <p>
      After polish, most of the error lay along outlines a few points couldn't follow: polish could move a point, never add one.
      So the polish now adds a point where an outline misses most, and keeps it if the error it saves pays for its coordinates.
      Fidelity rose on every set, but the outlines came out wiggly: they turned 1.5× as much as the clean SVG's. The new points
      had followed the corruption's noise, which a coordinate's price alone couldn't tell from a real bend.
    </p>
    <p>
      The fix was to charge for turning as well. A convex outline turns one full circle however many points it has, so the
      charge falls only on dents and zigzags. At 32 bits a radian, outlines turned within 5% of what they did before growth, and
      most of the gain stayed.
    </p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-4 py-2 font-medium">Held-out, scored once</th>
            <th class="px-4 py-2 text-right font-medium">Before</th>
            <th class="px-4 py-2 text-right font-medium">Growth, turning free</th>
            <th class="px-4 py-2 text-right font-medium">Turning priced</th>
          </tr>
        </thead>
        <tbody>
          <tr class="border-b border-line/60"><td class="px-4 py-2"><code>synthetic</code></td><td class="num px-4 py-2 text-right">34.67</td><td class="num px-4 py-2 text-right">34.71</td><td class="num px-4 py-2 text-right">34.72</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2"><code>icons</code></td><td class="num px-4 py-2 text-right">25.09</td><td class="num px-4 py-2 text-right">25.64</td><td class="num px-4 py-2 text-right">25.54</td></tr>
          <tr class="border-b border-line/60"><td class="px-4 py-2"><code>broad</code></td><td class="num px-4 py-2 text-right">27.95</td><td class="num px-4 py-2 text-right">28.44</td><td class="num px-4 py-2 text-right">28.37</td></tr>
          <tr><td class="px-4 py-2">turning, × the clean SVG's (<code>broad</code>)</td><td class="num px-4 py-2 text-right">0.96</td><td class="num px-4 py-2 text-right">1.48</td><td class="num px-4 py-2 text-right">1.00</td></tr>
        </tbody>
      </table>
    </div>
    <Callout variant="note" title="A score can't see everything">
      Fidelity alone would have kept the wiggly outlines: they scored higher. Smoothness had been measured in every report
      all along; it took looking at it to see what the extra points had bought.
    </Callout>

    <h2>Where it stands</h2>
    <p>
      Stage one of the plan aimed for 35.5 dB on training icons with no corruption. Shape and colour rules stopped at 34.84, the
      polish took it to 36.60, and growth to 37.95. The corrupted images are limited too: a warped copy of the clean SVG, scored against itself,
      reaches only 25.52 dB on <code>medium</code> and 21.06 on <code>heavy</code>.
    </p>
    <p>
      Layer recovery on icons is 0.63, against 0.89 on <code>synthetic</code>. A picture can match while its layers don't, and the
      layers are what someone editing the SVG wants, so that has a stage of its own. Learned parts are still ahead, and each
      will have to show the same thing the pieces here did. You can
      follow it on the <NuxtLink to="/games/copycat">game page</NuxtLink>, and each report is under
      <NuxtLink to="/runs">Runs</NuxtLink>.
    </p>

    <h2>Every version</h2>
    <p>
      The tracer is numbered: one version per kept change, so a score always names the algorithm that made it. The game
      page's leaderboard shows the latest. Held-out scores are clean fidelity in dB, taken once per change; a dash is a set
      that didn't exist yet. From version 14 on, each change has a recorded report under <NuxtLink to="/runs">Runs</NuxtLink>.
    </p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-3 py-2 font-medium">Version</th>
            <th class="px-3 py-2 font-medium">Change</th>
            <th class="px-3 py-2 text-right font-medium">synthetic</th>
            <th class="px-3 py-2 text-right font-medium">icons</th>
            <th class="px-3 py-2 text-right font-medium">broad</th>
            <th class="px-3 py-2 text-right font-medium">PR</th>
          </tr>
        </thead>
        <tbody class="num">
          <tr v-for="v in [...COPYCAT_VERSIONS].reverse()" :key="v.version" class="border-b border-line/60 last:border-0">
            <td class="px-3 py-1.5">{{ v.version }}</td>
            <td class="px-3 py-1.5 font-sans">{{ v.change }}</td>
            <td class="px-3 py-1.5 text-right">{{ score(v.heldout.synthetic) }}</td>
            <td class="px-3 py-1.5 text-right">{{ score(v.heldout.icons) }}</td>
            <td class="px-3 py-1.5 text-right">{{ score(v.heldout.broad) }}</td>
            <td class="px-3 py-1.5 text-right">
              <a :href="`https://github.com/0xMack/red-queen/pull/${v.pr}`" class="underline hover:text-fg" target="_blank" rel="noopener">#{{ v.pr }}</a>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </article>
</template>
