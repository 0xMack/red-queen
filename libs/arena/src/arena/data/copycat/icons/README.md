# Copy Cat `icons` set

`clean/` holds 152 emoji from [Twemoji](https://github.com/jdecked/twemoji) v17.0.3: graphics copyright 2019 Twitter,
Inc and other contributors, licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

They were changed:
- converted into Copy Cat's SVG subset (`arena.twemoji`): scaled from 36 to 128 px, group fills pushed down, and paths
  rewritten as absolute M/L/C/Q/Z, with arcs as cubics;
- `heldout/` holds 24 of them as `corrupt.v1` images (`arena.copycat_sets --set icons`), beside their clean SVGs.

The sample is single-codepoint emoji under 6 KB, drawn with the seed `copycat-icons-v1`. Eight of the 160 drawn were
left out because they use a transform or opacity.
