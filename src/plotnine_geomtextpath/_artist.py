"""Draw-time matplotlib Artist for curved/straight text along a path with line gap & optional label box."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.transforms import IdentityTransform

from ._geometry import (
    arc_lengths,
    curved_box_polygon,
    deduplicate_path,
    rounded_rect_polygon,
    sample_point_and_tangent,
    split_polyline_for_gap,
)

PT_PER_MM = 72.27 / 25.4
LW_PT_PER_MM = PT_PER_MM * (72.0 / 96.0)  # matches R grid lwd = mm * 72.27/25.4 in 1/96 in units


def _parse_fontface(fontface: Any) -> tuple[str, str]:
    if isinstance(fontface, (int, float, np.integer, np.floating)):
        code = int(fontface)
        return ("bold" if code in (2, 4) else "normal"), ("italic" if code in (3, 4) else "normal")
    s = str(fontface or "plain").strip().lower().replace("_", ".").replace(" ", ".")
    if s in ("bold.italic", "bolditalic", "italic.bold", "4"):
        return "bold", "italic"
    if s in ("bold", "2"):
        return "bold", "normal"
    if s in ("italic", "oblique", "3"):
        return "normal", "italic"
    return "normal", "normal"


def _parse_linetype(linetype: Any, lw_pt: float) -> tuple[float, tuple[float, ...] | None]:
    """Convert ggplot2/plotnine linetype specification into matplotlib gc.set_dashes(offset, seq) in points."""
    if linetype is None or linetype is False:
        return 0.0, None
    if isinstance(linetype, (tuple, list)):
        if len(linetype) == 2 and isinstance(linetype[1], (tuple, list)):
            return float(linetype[0]), tuple(float(v) for v in linetype[1])
        return 0.0, tuple(float(v) for v in linetype)
    s = str(linetype).strip().lower()
    scale = max(lw_pt, 1.0)
    presets: dict[str, tuple[float, ...] | None] = {
        "1": None,
        "solid": None,
        "-": None,
        "2": (4.0 * scale, 4.0 * scale),
        "dashed": (4.0 * scale, 4.0 * scale),
        "--": (4.0 * scale, 4.0 * scale),
        "3": (1.0 * scale, 3.0 * scale),
        "dotted": (1.0 * scale, 3.0 * scale),
        ":": (1.0 * scale, 3.0 * scale),
        "4": (1.0 * scale, 3.0 * scale, 4.0 * scale, 3.0 * scale),
        "dotdash": (1.0 * scale, 3.0 * scale, 4.0 * scale, 3.0 * scale),
        "dashdot": (1.0 * scale, 3.0 * scale, 4.0 * scale, 3.0 * scale),
        "-.": (1.0 * scale, 3.0 * scale, 4.0 * scale, 3.0 * scale),
        "5": (7.0 * scale, 3.0 * scale),
        "longdash": (7.0 * scale, 3.0 * scale),
        "6": (2.0 * scale, 2.0 * scale, 6.0 * scale, 2.0 * scale),
        "twodash": (2.0 * scale, 2.0 * scale, 6.0 * scale, 2.0 * scale),
    }
    if s in presets:
        return 0.0, presets[s]
    if s in ("0", "blank", "none", ""):
        return 0.0, ()
    return 0.0, None


def _to_rgba(color: Any, alpha: float) -> tuple[float, float, float, float] | None:
    if color is None:
        return None
    if isinstance(color, str) and color.strip().lower() in ("none", "na", "transparent", ""):
        return None
    try:
        r, g, b, a = mcolors.to_rgba(color)
        eff_a = a * alpha if (alpha is not None and np.isfinite(alpha)) else a
        if eff_a <= 0.0:
            return None
        return (float(r), float(g), float(b), float(min(1.0, max(0.0, eff_a))))
    except Exception:
        return None


class _TextPathArtist(Artist):
    """Draw-time artist that lays out curved/straight glyphs, gapped polylines, and label boxes in display px."""

    def __init__(
        self,
        ax,
        xy_data: np.ndarray,
        label: str,
        *,
        is_label: bool = False,
        text_only: bool = False,
        gap: bool | None = None,
        upright: bool = True,
        straight: bool = False,
        padding_in: float = 0.05,
        hjust: float = 0.5,
        vjust: float = 0.5,
        size_mm: float = 3.88,
        linewidth_mm: float = 0.5,
        linetype: Any = "solid",
        lineend: str = "butt",
        linejoin: str = "round",
        spacing: float = 0.0,
        family: str = "",
        fontface: Any = "plain",
        text_color: Any = "black",
        line_color: Any = "black",
        alpha: float = 1.0,
        fill: Any = "white",
        box_color: Any = None,
        box_linewidth_mm: float | None = None,
        box_linetype: Any = "solid",
        label_padding_lines: float = 0.25,
        label_r_lines: float = 0.15,
        zorder: float = 2.0,
    ):
        super().__init__()
        self._ax = ax
        self._xy_data = np.asarray(xy_data, dtype=float)
        self._label = str(label)
        self._is_label = bool(is_label)
        self._text_only = bool(text_only)
        self._gap = gap
        self._upright = bool(upright)
        self._straight = bool(straight)
        self._padding_in = float(padding_in)
        self._hjust = float(hjust)
        self._vjust = float(vjust)
        self._size_mm = float(size_mm)
        self._linewidth_mm = float(linewidth_mm)
        self._linetype = linetype
        self._lineend = str(lineend or "butt")
        self._linejoin = str(linejoin or "round")
        self._spacing = float(spacing)
        self._family = str(family or "")
        self._fontface = fontface
        self._text_color = text_color
        self._line_color = line_color
        self._alpha = float(alpha) if (alpha is not None and np.isfinite(alpha)) else 1.0
        self._fill = fill
        self._box_color = box_color
        self._box_linewidth_mm = box_linewidth_mm
        self._box_linetype = box_linetype
        self._label_padding_lines = float(label_padding_lines)
        self._label_r_lines = float(label_r_lines)
        self.set_zorder(zorder)
        self.set_clip_on(True)
        self.set_clip_box(ax.bbox)

    def _font_properties(self) -> FontProperties:
        weight, style = _parse_fontface(self._fontface)
        fontsize_pt = self._size_mm * PT_PER_MM
        fam = self._family.strip() if self._family else "DejaVu Sans"
        return FontProperties(family=fam, style=style, weight=weight, size=fontsize_pt)

    def draw(self, renderer):
        if not self.get_visible() or len(self._xy_data) < 2:
            return
        self.set_clip_box(self._ax.bbox)
        fig = self._ax.figure
        dpi = float(fig.dpi)
        _, fig_h = fig.canvas.get_width_height()
        fig_h = float(fig_h)

        # Transform path to isotropic display pixels so aspect ratio never distorts angles or spacing
        pts = deduplicate_path(self._ax.transData.transform(self._xy_data))
        if len(pts) < 2:
            return
        s_cum = arc_lengths(pts)
        L = float(s_cum[-1])
        if L <= 1e-6:
            return

        prop = self._font_properties()
        fontsize_pt = float(prop.get_size_in_points())
        fontsize_px = fontsize_pt * dpi / 72.0
        chars = list(self._label)
        n_chars = len(chars)

        # Measure per-glyph kerning-aware advances and body height
        advances = np.zeros(n_chars, dtype=float)
        glyph_widths = np.zeros(n_chars, dtype=float)
        prev_cum = 0.0
        for i, ch in enumerate(chars):
            w_ch, _, _ = renderer.get_text_width_height_descent(ch, prop, ismath=False)
            if ch == " " and w_ch <= 1e-3:
                w_ch = 0.32 * fontsize_px
            glyph_widths[i] = float(w_ch)
            prefix = self._label[: i + 1]
            # Append a sentinel when prefix ends in space so trailing space advance is measured
            if ch == " ":
                w_sent, _, _ = renderer.get_text_width_height_descent(prefix + "|", prop, ismath=False)
                w_bar, _, _ = renderer.get_text_width_height_descent("|", prop, ismath=False)
                cum = max(prev_cum + float(w_ch), float(w_sent) - float(w_bar))
            else:
                w_pref, _, _ = renderer.get_text_width_height_descent(prefix, prop, ismath=False)
                cum = float(w_pref)
            adv = cum - prev_cum
            advances[i] = adv if adv > 0.25 * float(w_ch) else float(w_ch)
            prev_cum = cum

        _, body_h_px, body_d_px = renderer.get_text_width_height_descent("Mg", prop, ismath=False)
        body_h_px = max(float(body_h_px), fontsize_px * 0.72)

        # Letter spacing in 1/1000 em units (calibrated to standard typographic tracking)
        extra_px = (self._spacing / 1000.0) * fontsize_px * 1.12
        total_w = float(np.sum(advances)) + max(0, n_chars - 1) * extra_px

        hjust = max(0.0, min(1.0, self._hjust))
        s_start = hjust * (L - total_w)
        s_end = s_start + total_w

        # Auto-flip reversed paths when upright=True so text reads left-to-right
        if self._upright and n_chars > 0:
            probe_s = np.linspace(max(0.0, s_start), min(L, s_end), max(3, min(11, n_chars)))
            txs = [sample_point_and_tangent(pts, s_cum, float(sq))[2] for sq in probe_s]
            if float(np.mean(txs)) < 0.0:
                pts = pts[::-1].copy()
                s_cum = arc_lengths(pts)
                # Keep label at the same physical position on the curve
                s_start_new = L - s_end
                s_end_new = L - s_start
                s_start, s_end = s_start_new, s_end_new

        # Normal offset from vjust: vjust=0.5 centers on line; vjust<0 lifts above; vjust>1 drops below
        normal_offset_px = (0.5 - self._vjust) * fontsize_px

        # Build offset curve for curved text so letter spacing stays uniform even on tight bends,
        # while anchoring the label midpoint to the exact base-curve hjust position.
        if not self._straight and abs(normal_offset_px) > 1e-3:
            off_pts = np.empty_like(pts)
            for k in range(len(pts)):
                _, _, tkx, tky = sample_point_and_tangent(pts, s_cum, float(s_cum[k]))
                off_pts[k, 0] = pts[k, 0] + normal_offset_px * (-tky)
                off_pts[k, 1] = pts[k, 1] + normal_offset_px * tkx
            # Keep 1-to-1 correspondence with s_cum so base-curve s_mid maps cleanly to offset arc length
            ds_off = np.hypot(np.diff(off_pts[:, 0]), np.diff(off_pts[:, 1]))
            s_text_cum = np.concatenate(([0.0], np.cumsum(ds_off)))
            if float(s_text_cum[-1]) > 1e-6:
                s_base_mid = 0.5 * (s_start + s_end)
                s_off_mid = float(np.interp(s_base_mid, s_cum, s_text_cum))
                s_text_start = s_off_mid - 0.5 * total_w
                text_pts = off_pts
                extra_norm_px = 0.0
            else:
                text_pts, s_text_cum, s_text_start, extra_norm_px = pts, s_cum, s_start, normal_offset_px
        else:
            text_pts, s_text_cum, s_text_start, extra_norm_px = pts, s_cum, s_start, normal_offset_px

        # Per-glyph arc-length centers along text_pts
        glyph_s = np.zeros(n_chars, dtype=float)
        cur = s_text_start
        for i in range(n_chars):
            glyph_s[i] = cur + 0.5 * advances[i]
            cur += advances[i] + extra_px

        # 1. Draw curve polylines (unless text_only)
        lw_pt = self._linewidth_mm * LW_PT_PER_MM
        line_rgba = _to_rgba(self._line_color, self._alpha)
        dash_off, dash_seq = _parse_linetype(self._linetype, lw_pt)
        is_blank_line = dash_seq is not None and len(dash_seq) == 0

        if not self._text_only and line_rgba is not None and lw_pt > 0.0 and not is_blank_line:
            if self._gap is None:
                use_gap = (not self._is_label) and (0.0 <= self._vjust <= 1.0)
            else:
                use_gap = bool(self._gap)

            if use_gap and n_chars > 0:
                pad_px = self._padding_in * dpi
                segs = split_polyline_for_gap(pts, s_cum, s_start - pad_px, s_end + pad_px)
            else:
                segs = [pts]

            for seg in segs:
                if len(seg) < 2:
                    continue
                gc = renderer.new_gc()
                gc.set_clip_rectangle(self._ax.bbox)
                gc.set_foreground(line_rgba)
                gc.set_alpha(line_rgba[3])
                gc.set_linewidth(lw_pt)
                gc.set_capstyle(self._lineend)
                gc.set_joinstyle(self._linejoin)
                if dash_seq is not None:
                    gc.set_dashes(dash_off, dash_seq)
                path = MplPath(seg)
                renderer.draw_path(gc, path, IdentityTransform(), rgbFace=None)
                gc.restore()

        if n_chars == 0:
            return

        # 2. Optional label background box (geom_labelpath / geom_labelline)
        if self._is_label:
            fill_rgba = _to_rgba(self._fill, self._alpha)
            bc = self._box_color if self._box_color is not None else self._line_color
            edge_rgba = _to_rgba(bc, self._alpha)
            box_lw_mm = self._box_linewidth_mm if self._box_linewidth_mm is not None else self._linewidth_mm
            box_lw_pt = float(box_lw_mm) * LW_PT_PER_MM
            pad_px = self._label_padding_lines * fontsize_px * 1.15
            rad_px = self._label_r_lines * fontsize_px * 1.15
            half_h = 0.5 * body_h_px + pad_px

            if self._straight:
                s_mid = 0.5 * (s_start + s_end)
                mx, my, tx, ty = sample_point_and_tangent(pts, s_cum, s_mid)
                nx, ny = -ty, tx
                theta = math.atan2(ty, tx)
                box_verts = rounded_rect_polygon(
                    mx + normal_offset_px * nx,
                    my + normal_offset_px * ny,
                    0.5 * total_w + pad_px,
                    half_h,
                    rad_px,
                    theta,
                )
            else:
                box_verts = curved_box_polygon(
                    pts, s_cum, s_start, s_end, normal_offset_px, half_h, pad_px, rad_px
                )

            closed_verts = np.vstack((box_verts, box_verts[0:1]))
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(box_verts) - 1) + [MplPath.CLOSEPOLY]
            box_path = MplPath(closed_verts, codes)
            gc = renderer.new_gc()
            gc.set_clip_rectangle(self._ax.bbox)
            if edge_rgba is not None and box_lw_pt > 0.0:
                gc.set_foreground(edge_rgba)
                gc.set_linewidth(box_lw_pt)
                boff, bseq = _parse_linetype(self._box_linetype, box_lw_pt)
                if bseq is not None and len(bseq) > 0:
                    gc.set_dashes(boff, bseq)
            else:
                gc.set_linewidth(0.0)
            if fill_rgba is not None:
                gc.set_alpha(fill_rgba[3])
            renderer.draw_path(gc, box_path, IdentityTransform(), rgbFace=fill_rgba)
            gc.restore()

        # 3. Draw each glyph along the path (or straight tangent)
        text_rgba = _to_rgba(self._text_color, self._alpha)
        if text_rgba is None:
            return

        if self._straight:
            s_mid = 0.5 * (s_start + s_end)
            mx, my, tx_m, ty_m = sample_point_and_tangent(pts, s_cum, s_mid)
            nx_m, ny_m = -ty_m, tx_m
            theta_m = math.atan2(ty_m, tx_m)
            ang_deg_m = math.degrees(theta_m)
            ca_m, sa_m = math.cos(theta_m), math.sin(theta_m)

        gc_t = renderer.new_gc()
        gc_t.set_clip_rectangle(self._ax.bbox)
        gc_t.set_foreground(text_rgba)
        gc_t.set_alpha(text_rgba[3])

        for i, ch in enumerate(chars):
            if self._straight:
                ds = float(glyph_s[i] - s_mid)
                cx = mx + ds * tx_m + normal_offset_px * nx_m
                cy = my + ds * ty_m + normal_offset_px * ny_m
                ca, sa, ang_deg = ca_m, sa_m, ang_deg_m
            else:
                px, py, tx, ty = sample_point_and_tangent(text_pts, s_text_cum, float(glyph_s[i]))
                nx, ny = -ty, tx
                cx = px + extra_norm_px * nx
                cy = py + extra_norm_px * ny
                theta = math.atan2(ty, tx)
                ca, sa, ang_deg = math.cos(theta), math.sin(theta), math.degrees(theta)

            # Place glyph so its rotated visual center lands exactly at (cx, cy)
            # Use consistent vertical baseline metric (body_h_px / body_d_px) so letters align cleanly on baseline
            lcx = 0.5 * float(glyph_widths[i])
            lcy = 0.5 * (body_h_px - 2.0 * body_d_px)
            x_bl = cx - (lcx * ca - lcy * sa)
            y_bl = cy - (lcx * sa + lcy * ca)
            y_td = fig_h - y_bl

            if ch == " ":
                # Skip inkless whitespace draw_text or emit? Emitting keeps exact string parity in draw calls.
                renderer.draw_text(gc_t, x_bl, y_td, " ", prop, ang_deg, ismath=False)
            else:
                renderer.draw_text(gc_t, x_bl, y_td, ch, prop, ang_deg, ismath=False)

        gc_t.restore()
