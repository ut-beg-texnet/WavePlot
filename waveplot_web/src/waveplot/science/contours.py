"""Contour lines at arbitrary levels.

``engine="contourpy"`` (default) uses the marching-squares implementation behind matplotlib.
``engine="v407"`` reproduces the original WavePlot scheme (DSPSNAP.PAS:232-366): every cell is
split into four triangles around its centre point (mean of the corners) and each triangle
contributes one straight segment per level (no line chaining).
"""

from __future__ import annotations

import numpy as np


def contour_lines(z: np.ndarray, x: np.ndarray, y: np.ndarray, levels, engine: str = "contourpy"
                  ) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return ``[(level, xs, ys)]`` with ``None``-free NaN separators between line pieces.

    ``z`` has shape ``(len(y), len(x))``.
    """
    z = np.asarray(z, dtype=float)
    out = []
    if engine == "v407":
        for lev in levels:
            xs, ys = _v407_segments(z, x, y, float(lev))
            out.append((float(lev), xs, ys))
        return out
    import contourpy

    gen = contourpy.contour_generator(x, y, np.ma.masked_invalid(z), name="serial",
                                      line_type=contourpy.LineType.SeparateCode)
    for lev in levels:
        pts, _codes = gen.lines(float(lev))
        if not pts:
            out.append((float(lev), np.empty(0), np.empty(0)))
            continue
        xs = np.concatenate([np.append(p[:, 0], np.nan) for p in pts])
        ys = np.concatenate([np.append(p[:, 1], np.nan) for p in pts])
        out.append((float(lev), xs, ys))
    return out


def _v407_segments(z: np.ndarray, x: np.ndarray, y: np.ndarray, lev: float) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(x, y)                       # (ny, nx)
    # corners: index = 2*xx + yy with xx along x, yy along y
    v = [z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:]]
    lo = np.minimum(np.minimum(v[0], v[1]), np.minimum(v[2], v[3]))
    hi = np.maximum(np.maximum(v[0], v[1]), np.maximum(v[2], v[3]))
    cells = np.nonzero((lo < lev) & (hi >= lev) | (lo <= lev) & (hi > lev))   # only cells the level crosses
    if cells[0].size == 0:
        return np.empty(0), np.empty(0)
    v = [a[cells] for a in v]
    px = [a[cells] for a in (X[:-1, :-1], X[1:, :-1], X[:-1, 1:], X[1:, 1:])]
    py = [a[cells] for a in (Y[:-1, :-1], Y[1:, :-1], Y[:-1, 1:], Y[1:, 1:])]
    v.append((v[0] + v[1] + v[2] + v[3]) / 4)
    px.append((px[0] + px[1] + px[2] + px[3]) / 4)
    py.append((py[0] + py[1] + py[2] + py[3]) / 4)
    segs_x, segs_y = [], []
    for a, b, c in ((0, 1, 4), (1, 3, 4), (0, 2, 4), (2, 3, 4)):
        pts_x, pts_y, hit = [], [], []
        for p, q in ((a, b), (b, c), (c, a)):
            vp, vq = v[p], v[q]
            cross = ((vp >= lev) & (lev > vq)) | ((vq >= lev) & (lev > vp))
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(cross, (lev - vp) / (vq - vp), 0.0)
            pts_x.append(px[p] + t * (px[q] - px[p]))
            pts_y.append(py[p] + t * (py[q] - py[p]))
            hit.append(cross)
        h = np.stack(hit)                            # (3, ncells)
        two = h.sum(axis=0) >= 2
        if not two.any():
            continue
        ex = np.stack(pts_x)
        ey = np.stack(pts_y)
        order = np.argsort(~h, axis=0, kind="stable")[:2]   # first two crossed edges
        i0 = np.take_along_axis(ex, order[:1], 0)[0][two]
        i1 = np.take_along_axis(ex, order[1:2], 0)[0][two]
        j0 = np.take_along_axis(ey, order[:1], 0)[0][two]
        j1 = np.take_along_axis(ey, order[1:2], 0)[0][two]
        segs_x.append(np.stack([i0, i1, np.full_like(i0, np.nan)], axis=1).ravel())
        segs_y.append(np.stack([j0, j1, np.full_like(j0, np.nan)], axis=1).ravel())
    if not segs_x:
        return np.empty(0), np.empty(0)
    return np.concatenate(segs_x), np.concatenate(segs_y)


def pick_contour_levels(levels: np.ndarray, max_lines: int = 24, skip_zero: bool = True) -> np.ndarray:
    """Subset of levels to draw (v407 drew at most 24, evenly picked)."""
    lev = np.asarray(levels, dtype=float)
    if skip_zero:
        lev = lev[lev != 0]
    if len(lev) <= max_lines:
        return lev
    idx = np.unique(np.linspace(0, len(lev) - 1, max_lines).round().astype(int))
    return lev[idx]
