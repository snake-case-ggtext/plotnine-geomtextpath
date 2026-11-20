"""Pure-numpy geometric utilities for curved text layout, path splitting, label boxes, and smoothing."""

from __future__ import annotations

import math
import numpy as np


def deduplicate_path(pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Drop consecutive duplicate or non-finite points from an (N, 2) polyline."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) <= 1:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    diffs = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    keep[1:] = diffs > eps
    return pts[keep]


def arc_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc length along an (N, 2) polyline."""
    if len(pts) == 0:
        return np.array([], dtype=float)
    if len(pts) == 1:
        return np.array([0.0], dtype=float)
    ds = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate(([0.0], np.cumsum(ds)))


def sample_point_and_tangent(pts: np.ndarray, s_cum: np.ndarray, s_query: float) -> tuple[float, float, float, float]:
    """Return (x, y, tx, ty) at arc length s_query along polyline (with smooth tangent extrapolation at ends)."""
    L = float(s_cum[-1])
    if len(pts) < 2 or L <= 1e-9:
        return float(pts[0, 0]), float(pts[0, 1]), 1.0, 0.0

    # Extrapolate linearly beyond ends if s_query falls outside [0, L]
    if s_query < 0.0:
        dx, dy = pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]
        norm = max(math.hypot(dx, dy), 1e-12)
        tx, ty = dx / norm, dy / norm
        return float(pts[0, 0] + s_query * tx), float(pts[0, 1] + s_query * ty), tx, ty
    if s_query > L:
        dx, dy = pts[-1, 0] - pts[-2, 0], pts[-1, 1] - pts[-2, 1]
        norm = max(math.hypot(dx, dy), 1e-12)
        tx, ty = dx / norm, dy / norm
        extra = s_query - L
        return float(pts[-1, 0] + extra * tx), float(pts[-1, 1] + extra * ty), tx, ty

    idx = int(np.searchsorted(s_cum, s_query, side="right")) - 1
    idx = max(0, min(idx, len(pts) - 2))
    seg_len = float(s_cum[idx + 1] - s_cum[idx])
    t = 0.0 if seg_len <= 1e-12 else (s_query - s_cum[idx]) / seg_len
    x = float(pts[idx, 0] + t * (pts[idx + 1, 0] - pts[idx, 0]))
    y = float(pts[idx, 1] + t * (pts[idx + 1, 1] - pts[idx, 1]))

    # Smooth central-difference tangent over a tiny arc window so sharp vertices don't jitter
    eps = min(2.0, L * 0.02)
    s_lo = max(0.0, s_query - eps)
    s_hi = min(L, s_query + eps)
    if s_hi - s_lo > 1e-6:
        x_lo = float(np.interp(s_lo, s_cum, pts[:, 0]))
        y_lo = float(np.interp(s_lo, s_cum, pts[:, 1]))
        x_hi = float(np.interp(s_hi, s_cum, pts[:, 0]))
        y_hi = float(np.interp(s_hi, s_cum, pts[:, 1]))
        dx, dy = x_hi - x_lo, y_hi - y_lo
    else:
        dx, dy = float(pts[idx + 1, 0] - pts[idx, 0]), float(pts[idx + 1, 1] - pts[idx, 1])
    norm = max(math.hypot(dx, dy), 1e-12)
    return x, y, dx / norm, dy / norm


def slice_polyline(pts: np.ndarray, s_cum: np.ndarray, s_a: float, s_b: float) -> np.ndarray | None:
    """Extract sub-polyline between arc lengths [s_a, s_b] with exact interpolated endpoints."""
    L = float(s_cum[-1])
    s_a = max(0.0, min(L, s_a))
    s_b = max(0.0, min(L, s_b))
    if s_b - s_a <= 1e-3:
        return None
    x_a = float(np.interp(s_a, s_cum, pts[:, 0]))
    y_a = float(np.interp(s_a, s_cum, pts[:, 1]))
    x_b = float(np.interp(s_b, s_cum, pts[:, 0]))
    y_b = float(np.interp(s_b, s_cum, pts[:, 1]))
    inner_mask = (s_cum > s_a + 1e-6) & (s_cum < s_b - 1e-6)
    inner = pts[inner_mask]
    if len(inner) > 0:
        out = np.vstack(([[x_a, y_a]], inner, [[x_b, y_b]]))
    else:
        out = np.array([[x_a, y_a], [x_b, y_b]], dtype=float)
    return out


def split_polyline_for_gap(pts: np.ndarray, s_cum: np.ndarray, s_lo: float, s_hi: float) -> list[np.ndarray]:
    """Split polyline into up to 2 sub-polylines excluding the interval (s_lo, s_hi)."""
    L = float(s_cum[-1])
    segs: list[np.ndarray] = []
    if s_lo > 1e-3:
        left = slice_polyline(pts, s_cum, 0.0, min(L, s_lo))
        if left is not None and len(left) >= 2:
            segs.append(left)
    if s_hi < L - 1e-3:
        right = slice_polyline(pts, s_cum, max(0.0, s_hi), L)
        if right is not None and len(right) >= 2:
            segs.append(right)
    return segs


def rounded_rect_polygon(
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    radius: float,
    angle_rad: float,
    arc_steps: int = 8,
) -> np.ndarray:
    """Build vertices of a rotated rounded rectangle centered at (cx, cy)."""
    r = max(0.0, min(radius, half_w * 0.95, half_h * 0.95))
    local_pts: list[tuple[float, float]] = []
    if r <= 1e-3:
        local_pts = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
    else:
        corners = [
            (half_w - r, -half_h + r, -math.pi / 2.0, 0.0),
            (half_w - r, half_h - r, 0.0, math.pi / 2.0),
            (-half_w + r, half_h - r, math.pi / 2.0, math.pi),
            (-half_w + r, -half_h + r, math.pi, 1.5 * math.pi),
        ]
        for ccx, ccy, a0, a1 in corners:
            for k in range(arc_steps + 1):
                a = a0 + (a1 - a0) * (k / arc_steps)
                local_pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))

    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    verts = np.array(
        [[cx + lx * ca - ly * sa, cy + lx * sa + ly * ca] for lx, ly in local_pts],
        dtype=float,
    )
    return verts


def curved_box_polygon(
    pts: np.ndarray,
    s_cum: np.ndarray,
    s_start: float,
    s_end: float,
    normal_offset: float,
    half_h: float,
    pad_x: float,
    radius: float,
    n_spine: int = 24,
    arc_steps: int = 6,
) -> np.ndarray:
    """Build a closed rounded ribbon polygon following the curve around [s_start - pad_x, s_end + pad_x]."""
    s0 = s_start - pad_x
    s1 = s_end + pad_x
    r = max(0.0, min(radius, half_h * 0.95, 0.45 * max(s1 - s0, 1e-3)))
    s_in0 = s0 + r
    s_in1 = s1 - r
    if s_in1 < s_in0:
        s_in0 = s_in1 = 0.5 * (s0 + s1)

    ss = np.linspace(s_in0, s_in1, max(4, n_spine))
    top_pts: list[tuple[float, float]] = []
    bot_pts: list[tuple[float, float]] = []
    for s in ss:
        x, y, tx, ty = sample_point_and_tangent(pts, s_cum, float(s))
        nx, ny = -ty, tx
        top_pts.append((x + (normal_offset + half_h) * nx, y + (normal_offset + half_h) * ny))
        bot_pts.append((x + (normal_offset - half_h) * nx, y + (normal_offset - half_h) * ny))

    # Rounded endcaps at s_in1 (right) and s_in0 (left)
    xr, yr, txr, tyr = sample_point_and_tangent(pts, s_cum, float(s_in1))
    nxr, nyr = -tyr, txr
    right_cap: list[tuple[float, float]] = []
    if r > 1e-3:
        # Top-right corner arc then bottom-right corner arc
        for k in range(1, arc_steps + 1):
            a = (math.pi / 2.0) * (1.0 - k / arc_steps)
            u = r * math.cos(a)
            v = (half_h - r) + r * math.sin(a)
            right_cap.append((xr + u * txr + (normal_offset + v) * nxr, yr + u * tyr + (normal_offset + v) * nyr))
        for k in range(1, arc_steps + 1):
            a = -(math.pi / 2.0) * (k / arc_steps)
            u = r * math.cos(a)
            v = -(half_h - r) + r * math.sin(a)
            right_cap.append((xr + u * txr + (normal_offset + v) * nxr, yr + u * tyr + (normal_offset + v) * nyr))

    xl, yl, txl, tyl = sample_point_and_tangent(pts, s_cum, float(s_in0))
    nxl, nyl = -tyl, txl
    left_cap: list[tuple[float, float]] = []
    if r > 1e-3:
        for k in range(1, arc_steps + 1):
            a = -math.pi / 2.0 - (math.pi / 2.0) * (k / arc_steps)
            u = r * math.cos(a)
            v = -(half_h - r) + r * math.sin(a)
            left_cap.append((xl + u * txl + (normal_offset + v) * nxl, yl + u * tyl + (normal_offset + v) * nyl))
        for k in range(1, arc_steps + 1):
            a = math.pi - (math.pi / 2.0) * (k / arc_steps)
            u = r * math.cos(a)
            v = (half_h - r) + r * math.sin(a)
            left_cap.append((xl + u * txl + (normal_offset + v) * nxl, yl + u * tyl + (normal_offset + v) * nyl))

    ring = top_pts + right_cap + list(reversed(bot_pts)) + left_cap
    return np.array(ring, dtype=float)


def loess_smooth(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray, span: float = 0.75, degree: int = 2) -> np.ndarray:
    """Tricube-weighted local polynomial regression (standard LOESS matching R's loess(y ~ x))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.interp(x_eval, x, y)
    k = max(degree + 2, min(n, int(math.ceil(span * n))))
    deg = min(degree, max(1, k - 2))
    out = np.empty(len(x_eval), dtype=float)
    for i, x0 in enumerate(x_eval):
        d = np.abs(x - x0)
        order = np.argsort(d)
        idx = order[:k]
        d_sub = d[idx]
        d_max = float(d_sub[-1])
        if d_max <= 1e-12:
            out[i] = float(np.mean(y[idx]))
            continue
        u = np.clip(d_sub / (d_max * 1.0001), 0.0, 1.0)
        w = (1.0 - u ** 3) ** 3
        dx = x[idx] - x0
        X = np.vstack([dx ** p for p in range(deg + 1)]).T
        W = w[:, None]
        XtW = (X * W).T
        lhs = XtW @ X
        rhs = XtW @ y[idx]
        try:
            beta = np.linalg.solve(lhs + np.eye(deg + 1) * 1e-12, rhs)
            out[i] = float(beta[0])
        except np.linalg.LinAlgError:
            out[i] = float(np.average(y[idx], weights=np.maximum(w, 1e-12)))
    return out
