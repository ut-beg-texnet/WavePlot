"""Grid reduction for display (``DI_ReadSnap`` QuickMax decimation and modern alternatives)."""

from __future__ import annotations

import numpy as np

from .compat import pas_round


def _block_edges(n: int, m: int) -> np.ndarray:
    """Edges splitting ``n`` samples into ``m`` near-equal contiguous blocks."""
    return np.linspace(0, n, m + 1).round().astype(int)


def fit_grid(a: np.ndarray, target: tuple[int, int], method: str = "mean") -> np.ndarray:
    """Reduce a 2-D array to at most ``target`` cells per axis.

    ``mean``: block average (NaN-aware). ``nearest``: point sampling of block centres. ``peak``:
    keep the value with the largest magnitude in each block (wavefronts stay visible).
    ``v407``: ``DI_ReadSnap`` with ``MapOn`` — trailing block averages ending at
    ``ROUND((p-1)·(n-1)/(m-1)+1)`` with the first sample kept exactly.
    """
    a = np.asarray(a)
    nx, ny = a.shape
    mx, my = min(target[0], nx), min(target[1], ny)
    if (mx, my) == (nx, ny):
        return a
    if method == "v407":
        return _v407_reduce(_v407_reduce(a, mx, axis=0), my, axis=1)
    if method == "nearest":
        ix = ((np.arange(mx) + 0.5) * nx / mx).astype(int)
        iy = ((np.arange(my) + 0.5) * ny / my).astype(int)
        return a[np.ix_(ix, iy)]
    ex, ey = _block_edges(nx, mx), _block_edges(ny, my)
    if method == "peak":
        out = np.empty((mx, my), dtype=a.dtype)
        for p in range(mx):
            rows = a[ex[p]:ex[p + 1]]
            for q in range(my):
                blk = rows[:, ey[q]:ey[q + 1]]
                k = np.nanargmax(np.abs(blk)) if np.isfinite(blk).any() else 0
                out[p, q] = blk.flat[k]
        return out
    # mean: cumulative sums give O(N) block means
    af = np.nan_to_num(a.astype(np.float64))
    w = np.isfinite(a).astype(np.float64)
    out = np.empty((mx, my))
    cnt = np.empty((mx, my))
    sx = np.add.reduceat(af, ex[:-1], axis=0)
    sw = np.add.reduceat(w, ex[:-1], axis=0)
    out[:] = np.add.reduceat(sx, ey[:-1], axis=1)
    cnt[:] = np.add.reduceat(sw, ey[:-1], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(cnt > 0, out / cnt, np.nan)
    return res.astype(a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32)


def _v407_reduce(a: np.ndarray, m: int, axis: int) -> np.ndarray:
    n = a.shape[axis]
    if m >= n:
        return a
    ratio = (n - 1) / (m - 1) if m > 1 else 0.0
    targets = [int(pas_round(p * ratio + 1)) - 1 for p in range(m)]   # 0-based inclusive ends
    parts = []
    prev = -1
    for t in targets:
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(prev + 1, t + 1) if t > prev else slice(t, t + 1)
        parts.append(np.take(a[tuple(sl)].mean(axis=axis, keepdims=True), 0, axis=axis))
        prev = t
    return np.stack(parts, axis=axis)


def coords_for(coord: np.ndarray, m: int, method: str = "mean") -> np.ndarray:
    """Coordinates matching :func:`fit_grid` output along one axis."""
    n = len(coord)
    if m >= n:
        return coord
    if method == "v407":
        ratio = (n - 1) / (m - 1) if m > 1 else 0.0
        return coord[[int(pas_round(p * ratio + 1)) - 1 for p in range(m)]]
    if method == "nearest":
        return coord[((np.arange(m) + 0.5) * n / m).astype(int)]
    e = _block_edges(n, m)
    return np.array([coord[e[p]:e[p + 1]].mean() for p in range(m)])


def dump_glyph_indices(n: int, m: int, mode: str = "standard") -> np.ndarray:
    """Indices of glyph positions along an axis of length ``n`` reduced to ``m`` (≤ n).

    v407 (``DI_ReadDump2``): ``ROUND((p - 0.49)·n/m)`` for p = 1..m (1-based).
    Standard: centres of ``m`` equal blocks.
    """
    m = min(m, n)
    if mode == "v407":
        ratio = n / m
        return np.clip(np.array([int(pas_round((p - 0.49) * ratio)) - 1 for p in range(1, m + 1)]), 0, n - 1)
    return ((np.arange(m) + 0.5) * n / m).astype(int)
