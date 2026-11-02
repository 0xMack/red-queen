# Copy Cat `broad` set

`clean/` holds flat-colour icons from two families, each file named for its own:

- `twemoji-*`: emoji from [Twemoji](https://github.com/jdecked/twemoji) v17.0.3: graphics copyright 2019 Twitter, Inc and
  other contributors, licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). A sample of the
  single-codepoint emoji under 6 KB, none of them in the `icons` set, drawn with the seed `copycat-broad-v1`.
- `flat-*`: [Flat Color Icons](https://github.com/icons8/flat-color-icons) at commit `1bf90d5` by Icons8, used under the
  MIT licence it is offered under: every icon in `svg/` that converts.

They were changed:
- converted into Copy Cat's SVG subset (`arena.twemoji`): scaled to 128 px, group fills pushed down, and paths rewritten as
  absolute M/L/C/Q/Z, with arcs as cubics;
- `heldout/` holds 120 of them as `corrupt.v1` images (`arena.copycat_sets --set broad`), beside their clean SVGs.

A file using a transform, a stroke, opacity or rounded rectangles is left out rather than approximated.
