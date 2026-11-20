"""plotnine-geomtextpath: curved text along paths, lines, smooths, and label boxes for plotnine."""

from ._geoms import (
    geom_labelline,
    geom_labelpath,
    geom_textline,
    geom_textpath,
    geom_textsmooth,
    stat_textsmooth,
)

__all__ = [
    "geom_textpath",
    "geom_textline",
    "geom_textsmooth",
    "geom_labelpath",
    "geom_labelline",
    "stat_textsmooth",
]
__version__ = "0.1.0"
