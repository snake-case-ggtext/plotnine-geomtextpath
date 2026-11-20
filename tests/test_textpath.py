"""Unit tests for plotnine-geomtextpath."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from plotnine import aes, coord_flip, facet_wrap, geom_point, ggplot

from plotnine_geomtextpath import (
    geom_labelline,
    geom_labelpath,
    geom_textline,
    geom_textpath,
    geom_textsmooth,
)
from plotnine_geomtextpath._geometry import (
    arc_lengths,
    loess_smooth,
    sample_point_and_tangent,
    split_polyline_for_gap,
)


def _draw_fig(p):
    fig = p.draw()
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig


def test_geometry_arc_and_split():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    s = arc_lengths(pts)
    assert np.allclose(s, [0.0, 10.0, 20.0])
    x, y, tx, ty = sample_point_and_tangent(pts, s, 5.0)
    assert math.isclose(x, 5.0, abs_tol=1e-6)
    assert math.isclose(tx, 1.0, abs_tol=1e-6)
    segs = split_polyline_for_gap(pts, s, 6.0, 14.0)
    assert len(segs) == 2
    assert math.isclose(segs[0][-1, 0], 6.0, abs_tol=1e-6)
    assert math.isclose(segs[1][0, 0], 14.0, abs_tol=1e-6)


def test_loess_smooth_recovers_sine():
    x = np.linspace(0, 2 * np.pi, 40)
    y = np.sin(x)
    x_eval = np.linspace(0, 2 * np.pi, 50)
    y_hat = loess_smooth(x, y, x_eval, span=0.6, degree=2)
    assert np.max(np.abs(y_hat - np.sin(x_eval))) < 0.08


def test_render_all_geoms():
    x = np.linspace(0, 2 * np.pi, 30)
    df = pd.DataFrame({
        "x": np.concatenate([x, x]),
        "y": np.concatenate([np.sin(x), np.cos(x)]),
        "grp": ["Sine"] * 30 + ["Cosine"] * 30,
    })
    for layer in (
        geom_textline(),
        geom_textpath(upright=True, text_only=True),
        geom_textpath(straight=True),
        geom_labelpath(fill="lightyellow", boxcolour="navy"),
        geom_labelline(straight=True, fill="white"),
        geom_textsmooth(method="lm"),
        geom_textsmooth(method="loess"),
    ):
        p = ggplot(df, aes("x", "y", color="grp", label="grp")) + layer
        fig = _draw_fig(p)
        assert fig.canvas.get_width_height()[0] > 0


def test_facets_and_coord_flip():
    x = np.linspace(0, 2 * np.pi, 25)
    df = pd.DataFrame({
        "x": np.concatenate([x, x]),
        "y": np.concatenate([np.sin(x), np.cos(x)]),
        "lbl": ["A curve"] * 25 + ["B curve"] * 25,
        "panel": ["P1"] * 25 + ["P2"] * 25,
    })
    fig1 = _draw_fig(ggplot(df, aes("x", "y", label="lbl")) + geom_textline() + facet_wrap("~panel"))
    assert len(fig1.axes) == 2

    df1 = df[df["panel"] == "P1"]
    fig2 = _draw_fig(ggplot(df1, aes("x", "y", label="lbl")) + geom_textline() + coord_flip())
    assert len(fig2.axes) == 1
