<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import * as code from "~/data/snippets/tracing-images"
import * as math from "~/data/math/tracing-images"
</script>

<template>
  <article class="prose-chapter">
    <p>
      Image generators are good at ideas for logos and icons, but what they hand back can't go on a page as an icon: a raster
      with no transparency, soft edges, grain, compression artifacts, and a background that is almost empty. What you want is
      the design the image depicts -- a small SVG with a transparent background, a few flat colours, and whole parts in
      layers, so a circle behind a square is a circle and not a crescent.
    </p>
    <p>
      This chapter explains the pipeline that gets there. Nothing in it is trained: it is one objective, applied at every
      step. Start with what it does:
    </p>

    <TraceLab />

    <h2>Likeness is the wrong score</h2>
    <p>
      A tracer such as VTracer or Potrace reproduces the pixels faithfully. For a generated image that is the wrong goal,
      because it reproduces the noise too: edges that follow the blur, speckles, dozens of near-identical colours, and regions
      cut out of each other instead of stacked.
    </p>
    <p>
      So a score of likeness to the input would reward exactly what we don't want. Copy Cat's benchmark starts from the other
      end: clean SVGs, rendered and then corrupted the way a generator would, with blur, noise, JPEG, colour drift, a slight
      warp and a background. An entrant only sees the corrupted image, and is scored against the clean render it never saw:
    </p>
    <MathFormula :formula="math.fidelity" />
    <p>
      PSNR is measured in decibels, which are logarithmic: every extra 10 dB means a tenth of the squared error. Averaging over
      black and over white scores colour and transparency together, so a leftover background loses, and so does a hole.
      Reproducing the noise loses too, because the clean render has none.
    </p>

    <h2>The shortest description</h2>
    <p>
      The pipeline's objective is minimum description length: find the shortest SVG that explains the image, where whatever
      the shapes leave unexplained has to be written down as well. A shape's length is the bits to say what it is, plus the bits
      to correct what it gets wrong. Move the slider:
    </p>

    <CodeLengthDemo />

    <p>
      The design priors you would otherwise have to write as rules come out of that one sum. A circle is cheaper than the
      four Béziers that draw it, so near-circles snap to circles. A straight run is one segment. A reused colour is cheaper
      than a new one, so the palette stays small. And noise isn't worth a shape, because a speck costs more to describe than
      the few pixels it would correct, so nothing has to say "denoise".
    </p>

    <h2>From pixels to parts</h2>
    <p>Before there are shapes to compare, the image has to become parts. In order:</p>
    <ol>
      <li><strong>Noise.</strong> The median colour difference between neighbouring pixels. Being a median, it reads the flat areas and ignores the edges. It sets how hard outlines may chase detail.</li>
      <li><strong>Background.</strong> A flat colour or a gradient, whichever is shorter, fitted to the image's border. It stays transparent in the output.</li>
      <li><strong>Palette.</strong> k-means in OKLab, a colour space where distance tracks what people see. Two colours merge while the error that adds costs less than the palette entry and the boundary it saves.</li>
      <li><strong>Regions.</strong> A majority filter, then specks and thin anti-aliased bands dissolve into what surrounds them.</li>
      <li><strong>Parts.</strong> One per connected region, each filled with the average colour of its interior. Two pieces of one colour join if one shape says them more briefly, unless they sit on another part, as two eyes sit on a face.</li>
    </ol>
    <CodeBlock :snippet="code.pipeline" />

    <h2>Stacked, not cut out</h2>
    <p>
      A tracer outputs the regions it sees, so a part behind another comes back with a bite out of it. Here, shapes have a paint
      order and may extend under the shapes above them, and the objective handles the rest: a pixel under an occluder costs
      nothing either way, so the shortest description of "a circle behind a square" is the whole circle.
    </p>
    <p>
      The order comes from the same fits. Each part is fitted with every other part free to lie over it. Of two touching parts,
      the one whose shape covers more of the other, as a fraction of it, goes below. Each part is then fitted for real, with
      only the parts above it free.
    </p>
    <Callout variant="note" title="How that gets measured">
      Layer recovery renders each clean part alone, hidden portions included, and finds its best match among the output's shapes,
      each also rendered alone. It is high only if the parts were separated and their hidden portions completed: the pipeline
      scores 0.89 on the held-out <code>synthetic</code> images, against VTracer's 0.73.
    </Callout>

    <h2>Choosing a shape</h2>
    <p>
      A part gets several candidate descriptions: circle, ellipse, rectangle, rounded rectangle, polygons and Bézier paths at a
      few tolerances, and smooth splines through a few knots. A spline's tangents follow from its neighbouring knots, so a blob
      costs its knots and not the cubics that draw it. The shortest candidates are then refined, one number at a time, for as
      long as the description keeps getting shorter.
    </p>
    <CodeBlock :snippet="code.choose" />
    <p>
      The final shape can then have holes cut in it, each one kept only while the whole gets shorter. So a ring is a disc with a
      reversed subpath inside it, and the subset of SVG the benchmark accepts can draw it.
    </p>

    <h2>Where the objective wasn't enough</h2>
    <p>
      Description length decides a lot, but not everything, and it would be wrong to pretend otherwise. The pixels often can't
      tell two explanations apart, and then a prior has to:
    </p>
    <ul>
      <li>
        <strong>What is the background?</strong> A tile filling most of the border and a background describe the image equally
        well. Only which one ends up transparent differs. The pipeline assumes the background reaches the corners, which is
        wrong when the subject owns two of them: a locomotive reaching both bottom corners dropped from 21.0 to 15.8 dB when
        that rule went in.
      </li>
      <li>
        <strong>Is it one colour or two?</strong> A part's own shading, which a flat fill can't say anyway, would otherwise survive
        as a second colour. <code>merge_delta</code> forces those together, and it also joins some colours an artist meant to differ.
      </li>
      <li>
        <strong>Is it clutter or a pale part?</strong> <code>background_delta</code> absorbs faint clutter into the background. A
        separate test gives a light region back if it stands out from the background and shares a quarter of its boundary with other parts.
      </li>
    </ul>
    <p>
      Each of those is a threshold someone tuned, with failure cases on record.
      <NuxtLink to="/learn/earning-complexity">Earning Each Piece of Complexity</NuxtLink> goes through how they, and everything
      else, got in.
    </p>

    <h2>How it does</h2>
    <p>On the 24 held-out <code>synthetic</code> images, six per corruption level:</p>
    <div class="card not-prose my-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line text-left label">
            <th class="px-4 py-2 font-medium">Entrant</th>
            <th class="px-4 py-2 text-right font-medium">Fidelity</th>
            <th class="px-4 py-2 text-right font-medium">clean</th>
            <th class="px-4 py-2 text-right font-medium">heavy</th>
            <th class="px-4 py-2 text-right font-medium">Layer recovery</th>
            <th class="px-4 py-2 text-right font-medium">Shapes vs. clean</th>
            <th class="px-4 py-2 text-right font-medium">Per image</th>
          </tr>
        </thead>
        <tbody class="num">
          <tr class="border-b border-line/60">
            <td class="px-4 py-2 font-sans">The pipeline</td>
            <td class="px-4 py-2 text-right">32.47 dB</td>
            <td class="px-4 py-2 text-right">45.30</td>
            <td class="px-4 py-2 text-right">23.02</td>
            <td class="px-4 py-2 text-right">0.89</td>
            <td class="px-4 py-2 text-right">0.99×</td>
            <td class="px-4 py-2 text-right">322 ms</td>
          </tr>
          <tr>
            <td class="px-4 py-2 font-sans">VTracer, background deleted</td>
            <td class="px-4 py-2 text-right">20.18 dB</td>
            <td class="px-4 py-2 text-right">28.25</td>
            <td class="px-4 py-2 text-right">11.71</td>
            <td class="px-4 py-2 text-right">0.73</td>
            <td class="px-4 py-2 text-right">6.67×</td>
            <td class="px-4 py-2 text-right">8 ms</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p>
      The pipeline is about 40 times slower, and still fast enough to run in your browser. Those images flatter it, though:
      <code>synthetic</code> is random stacks of primitives and blobs, which suit its candidates. On the held-out
      <code>icons</code>, real Twemoji with thin details and many parts, it scores 23.68 dB, and those icons take about 4 seconds
      each. The <NuxtLink to="/games/copycat">game page</NuxtLink> has the full field, and you can drop in your own image there.
    </p>
  </article>
</template>
