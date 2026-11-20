"""plotnine geoms and stats for geomtextpath: geom_textpath, geom_textline, geom_textsmooth,
geom_labelpath, and geom_labelline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from plotnine.geoms.geom import geom
from plotnine.geoms.geom_text import geom_text
from plotnine.stats.stat import stat
from scipy import stats as sp_stats

from ._artist import _TextPathArtist
from ._geometry import loess_smooth

_ALIAS_MAP = {
    "colour": "color",
    "linecolour": "linecolor",
    "textcolour": "textcolor",
    "boxcolour": "boxcolor",
    "label.padding": "label_padding",
    "label.r": "label_r",
}


def _normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        nk = _ALIAS_MAP.get(k, k)
        out[nk] = v
    return out


def _first_valid(series: pd.Series, default: Any) -> Any:
    if series is None or len(series) == 0:
        return default
    for v in series:
        if v is None:
            continue
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            continue
        if pd.isna(v):
            continue
        return v
    return default


def _parse_padding_in(val: Any, default: float = 0.05) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, (tuple, list)) and len(val) == 2:
        num, unit = float(val[0]), str(val[1]).lower().strip()
        if unit in ("in", "inch", "inches"):
            return num
        if unit in ("pt", "pts", "bigpts", "points"):
            return num / 72.0
        if unit in ("mm",):
            return num / 25.4
        if unit in ("cm",):
            return num / 2.54
    return default


class stat_textsmooth(stat):
    """Smooth stat for geom_textsmooth supporting lm and tricube LOESS while preserving label aesthetics."""

    REQUIRED_AES = {"x", "y"}
    DEFAULT_PARAMS = {
        "geom": "textpath",
        "position": "identity",
        "na_rm": False,
        "method": "auto",
        "formula": None,
        "se": True,
        "level": 0.95,
        "n": 80,
        "span": 0.75,
        "fullrange": False,
    }

    def compute_group(self, data: pd.DataFrame, scales) -> pd.DataFrame:
        df = data.dropna(subset=["x", "y"]).sort_values("x")
        if len(df) < 2:
            return pd.DataFrame()

        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        n = len(x)
        n_out = int(self.params.get("n", 80))
        x_min, x_max = float(x.min()), float(x.max())
        if self.params.get("fullrange") and scales is not None and getattr(scales, "x", None) is not None:
            lim = scales.x.dimension()
            x_min, x_max = float(lim[0]), float(lim[1])
        x_eval = np.linspace(x_min, x_max, max(2, n_out))

        method = self.params.get("method", "auto")
        if method in (None, "auto"):
            method = "loess" if n < 1000 else "lm"
        method_s = str(method).lower()
        se_flag = bool(self.params.get("se", True))
        level = float(self.params.get("level", 0.95))

        se_fit = None
        df_resid = max(1, n - 2)
        if method_s == "lm":
            slope, intercept = np.polyfit(x, y, 1)
            y_eval = intercept + slope * x_eval
            if se_flag and n > 2:
                resid = y - (intercept + slope * x)
                s2 = float(np.sum(resid ** 2)) / (n - 2)
                x_bar = float(np.mean(x))
                sxx = max(float(np.sum((x - x_bar) ** 2)), 1e-12)
                se_fit = np.sqrt(s2 * (1.0 / n + (x_eval - x_bar) ** 2 / sxx))
                df_resid = n - 2
        elif method_s in ("loess", "lowess"):
            span = float(self.params.get("span", 0.75))
            y_eval = loess_smooth(x, y, x_eval, span=span, degree=2)
            if se_flag and n > 4:
                y_fit = loess_smooth(x, y, x, span=span, degree=2)
                resid = y - y_fit
                df_resid = max(2, int(round(n - 3.0 / max(span, 0.25))))
                s_hat = float(np.sqrt(np.sum(resid ** 2) / df_resid))
                k_eff = max(3.0, span * n)
                se_fit = np.full(len(x_eval), s_hat / np.sqrt(k_eff), dtype=float)
        else:
            deg = 2 if method_s in ("poly", "quadratic") else 1
            coeffs = np.polyfit(x, y, deg)
            y_eval = np.polyval(coeffs, x_eval)

        out = pd.DataFrame({"x": x_eval, "y": y_eval})
        if se_flag and se_fit is not None:
            t_crit = float(sp_stats.t.ppf(0.5 + 0.5 * level, df_resid))
            out["se"] = se_fit
            out["ymin"] = y_eval - t_crit * se_fit
            out["ymax"] = y_eval + t_crit * se_fit
        # Preserve group-level non-positional aesthetics (label, color, hjust, vjust, etc.)
        for col in df.columns:
            if col in ("x", "y"):
                continue
            vals = df[col].dropna().unique()
            if len(vals) == 1:
                out[col] = vals[0]
            elif col in ("group", "PANEL", "label"):
                out[col] = df[col].iloc[0]
        return out


class geom_textpath(geom):
    """Place curved (or straight) text directly along an arbitrary path."""

    DEFAULT_AES = {
        "color": "black",
        "alpha": 1.0,
        "size": 3.88,
        "hjust": 0.5,
        "vjust": 0.5,
        "family": "",
        "fontface": "plain",
        "lineheight": 1.2,
        "linewidth": 0.5,
        "linetype": "solid",
        "spacing": 0.0,
        "linecolor": None,
        "textcolor": None,
        "angle": 0.0,
    }
    REQUIRED_AES = {"x", "y", "label"}
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "lineend": "butt",
        "linejoin": "round",
        "linemitre": 10,
        "text_only": False,
        "gap": None,
        "upright": True,
        "halign": "center",
        "offset": None,
        "straight": False,
        "padding": 0.05,
        "text_smoothing": 0,
        "remove_long": False,
        "rich": False,
        "arrow": None,
        "sort_by_x": False,
        "is_label": False,
    }

    def __init__(self, mapping=None, data=None, **kwargs):
        kwargs = _normalize_kwargs(kwargs)
        super().__init__(mapping, data, **kwargs)

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params, coord, ax, params: dict[str, Any]):
        if data is None or len(data) < 2:
            return
        df = data.copy()
        if params.get("sort_by_x"):
            df = df.sort_values("x", kind="mergesort")
        df = coord.transform(df, panel_params)
        df = df.dropna(subset=["x", "y"])
        if len(df) < 2:
            return

        xy = df[["x", "y"]].to_numpy(dtype=float)
        label = str(_first_valid(df["label"], "")) if "label" in df.columns else ""
        base_color = _first_valid(df["color"], "black") if "color" in df.columns else "black"
        text_color = _first_valid(df["textcolor"], None) if "textcolor" in df.columns else None
        line_color = _first_valid(df["linecolor"], None) if "linecolor" in df.columns else None
        if text_color is None:
            text_color = base_color
        if line_color is None:
            line_color = base_color

        alpha = float(_first_valid(df["alpha"], 1.0)) if "alpha" in df.columns else 1.0
        size_mm = float(_first_valid(df["size"], 3.88)) if "size" in df.columns else 3.88
        linewidth_mm = float(_first_valid(df["linewidth"], 0.5)) if "linewidth" in df.columns else 0.5
        linetype = _first_valid(df["linetype"], "solid") if "linetype" in df.columns else "solid"
        hjust = float(_first_valid(df["hjust"], 0.5)) if "hjust" in df.columns else 0.5
        vjust = float(_first_valid(df["vjust"], 0.5)) if "vjust" in df.columns else 0.5
        spacing = float(_first_valid(df["spacing"], 0.0)) if "spacing" in df.columns else 0.0
        family = str(_first_valid(df["family"], "")) if "family" in df.columns else ""
        fontface = _first_valid(df["fontface"], "plain") if "fontface" in df.columns else "plain"

        is_label = bool(params.get("is_label", False))
        fill = _first_valid(df["fill"], "white") if (is_label and "fill" in df.columns) else "white"
        box_color = _first_valid(df["boxcolor"], None) if (is_label and "boxcolor" in df.columns) else None
        box_lw = _first_valid(df["boxlinewidth"], None) if (is_label and "boxlinewidth" in df.columns) else None
        box_lt = _first_valid(df["boxlinetype"], "solid") if (is_label and "boxlinetype" in df.columns) else "solid"

        artist = _TextPathArtist(
            ax,
            xy,
            label,
            is_label=is_label,
            text_only=bool(params.get("text_only", False)),
            gap=params.get("gap", None),
            upright=bool(params.get("upright", True)),
            straight=bool(params.get("straight", False)),
            padding_in=_parse_padding_in(params.get("padding", 0.05)),
            hjust=hjust,
            vjust=vjust,
            size_mm=size_mm,
            linewidth_mm=linewidth_mm,
            linetype=linetype,
            lineend=str(params.get("lineend", "butt")),
            linejoin=str(params.get("linejoin", "round")),
            spacing=spacing,
            family=family,
            fontface=fontface,
            text_color=text_color,
            line_color=line_color,
            alpha=alpha,
            fill=fill,
            box_color=box_color,
            box_linewidth_mm=float(box_lw) if box_lw is not None else None,
            box_linetype=box_lt,
            label_padding_lines=float(params.get("label_padding", 0.25)),
            label_r_lines=float(params.get("label_r", 0.15)),
            zorder=float(params.get("zorder", 2.0)),
        )
        ax.add_artist(artist)

    @staticmethod
    def draw_legend(data: pd.Series, da, lyr):
        row = data.copy()
        row["label"] = "a"
        row["angle"] = 0
        row["size"] = float(row.get("size", 3.88)) * 2.845
        return geom_text.draw_legend(row, da, lyr)


class geom_textline(geom_textpath):
    """Place curved (or straight) text along an x-ordered line."""

    DEFAULT_PARAMS = {
        **geom_textpath.DEFAULT_PARAMS,
        "sort_by_x": True,
    }


class geom_textsmooth(geom_textpath):
    """Fit a linear or locally weighted smooth curve with curved text."""

    DEFAULT_PARAMS = {
        **geom_textpath.DEFAULT_PARAMS,
        "stat": stat_textsmooth,
        "sort_by_x": True,
        "method": "auto",
        "formula": None,
        "se": True,
        "level": 0.95,
        "n": 80,
        "span": 0.75,
        "fullrange": False,
    }


class geom_labelpath(geom_textpath):
    """Place curved (or straight) text inside a rounded background box along a path."""

    DEFAULT_AES = {
        **geom_textpath.DEFAULT_AES,
        "fill": "white",
        "boxcolor": None,
        "boxlinewidth": None,
        "boxlinetype": "solid",
    }
    DEFAULT_PARAMS = {
        **geom_textpath.DEFAULT_PARAMS,
        "gap": False,
        "is_label": True,
        "label_padding": 0.25,
        "label_r": 0.15,
    }


class geom_labelline(geom_labelpath):
    """Place curved (or straight) text inside a rounded background box along an x-ordered line."""

    DEFAULT_PARAMS = {
        **geom_labelpath.DEFAULT_PARAMS,
        "sort_by_x": True,
    }
