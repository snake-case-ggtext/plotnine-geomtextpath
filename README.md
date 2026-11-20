# plotnine-geomtextpath

Curved and straight text along arbitrary paths, lines, and smoothed curves for [plotnine](https://plotnine.org/), inspired by R's `geomtextpath`.

## Features

- `geom_textpath()` — place curved (or `straight=True`) text along arbitrary ordered paths
- `geom_textline()` — place curved text along x-ordered lines
- `geom_textsmooth()` — fit linear (`method="lm"`) or locally weighted (`method="loess"`) smooths with curved text labels
- `geom_labelpath()` and `geom_labelline()` — curved or straight text inside rounded background label boxes
- Aspect-ratio-invariant draw-time layout in display pixels, automatic upright text flipping (`upright=True`), automatic line gaps behind text (`gap`, `padding`), letter `spacing`, `hjust`/`vjust`, `text_only`, separate `textcolor`/`linecolor`/`boxcolour`/`fill`, `linetype`, `linewidth`, `fontface`, facets, and `coord_flip()`.
