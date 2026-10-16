"""Colour / contour level distributions.

``mode="v407"`` transliterates ``CalcLevels`` and ``GetLevel`` (DSPSNAP.PAS:32-205) exactly,
including the range shrink, the cosine overshoot, the zero-level rule and band-toward-zero.
``mode="standard"`` uses the exact data range and well-behaved distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .compat import Mode, Provenance, pas_trunc

Method = Literal["lin", "quad", "dquad", "cos", "fixed", "log"]
METHODS: dict[str, str] = {
    "lin": "Linear",
    "quad": "Quadratic (x²)",
    "dquad": "Double quadratic (x⁴)",
    "cos": "Cosine",
    "fixed": "Fixed table",
    "log": "Symmetric logarithmic",
}
LEV_ARRAY = (0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.79, 0.86, 0.91, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 1)


@dataclass
class Levels:
    values: np.ndarray               # level values, length n (ascending for monotone methods)
    method: str
    vmin: float                      # range the levels were computed from
    vmax: float
    abs_values: bool = False
    zero_index: int | None = None    # index (0-based) of the level forced to exactly 0
    ratio: float | None = None       # log distribution: ratio between successive levels
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def n(self) -> int:
        return len(self.values)

    def band_index(self, z: np.ndarray, rule: str | None = None) -> np.ndarray:
        """Level index (0-based) of each value.

        ``toward_zero`` (v407 ``GetLevel``): positive values take the largest level ≤ v, negative
        values the smallest level ≥ v. ``floor``: largest level ≤ v everywhere.
        """
        rule = rule or ("toward_zero" if self.provenance.mode == "v407" else "floor")
        z = np.asarray(z, dtype=float)
        if self.abs_values:
            z = np.abs(z)
        lev = np.asarray(self.values, dtype=float)
        n = len(lev)
        floor_idx = np.clip(np.searchsorted(lev, z, side="right") - 1, 0, n - 1)
        if rule == "floor":
            return floor_idx
        ceil_idx = np.clip(np.searchsorted(lev, z, side="left"), 0, n - 1)
        return np.where(z >= 0, floor_idx, ceil_idx)


def _shrink(vmax: float, vmin: float, n: int) -> tuple[float, float]:
    rat = (n - 0.5) / n
    vmax = vmax * rat if vmax >= 0 else vmax / rat
    vmin = vmin * rat if vmin <= 0 else vmin / rat
    return vmax, vmin


def calc_one_level_ratio(vmax: float, vmin: float, n: int, n_range: float) -> float:
    """``Calc1Lev``: per-level ratio of the symmetric log distribution."""
    zz = max(abs(vmax), abs(vmin))
    z = min(abs(vmax), abs(vmin))
    zs = zz / n_range
    if zs >= z or vmax < 0 or vmin > 0:
        return math.exp(math.log(n_range) / n)
    a = math.log(zz / z) / math.log(z / zs)
    nn = pas_trunc((n - 1) / (a + 2))
    m = n - 2 * nn - 1
    return math.exp(math.log(zz / zs) / max(m + nn, 1))


def _log_levels_v407(vmax: float, vmin: float, n: int, ratio: float) -> tuple[np.ndarray, int]:
    lev = np.zeros(n + 2)  # 1-based like Pascal
    if vmin == 0:
        lev0 = 1
    elif vmax == 0:
        lev0 = n
    else:
        lev0 = pas_trunc((n - math.log(abs(vmax / vmin)) / math.log(ratio)) / 2) + 1
    if lev0 > n or vmax < 0:
        lev0 = n
    elif lev0 <= 0 or vmin > 0:
        lev0 = 1
    if abs(vmin) > abs(vmax):
        lev[1] = vmin
        for k in range(2, lev0):
            lev[k] = lev[k - 1] / ratio
        if lev0 < n:
            lev[lev0 + 1] = -lev[lev0 - 1]
            for k in range(lev0 + 2, n + 1):
                lev[k] = lev[k - 1] * ratio
    else:
        lev[n] = vmax
        for k in range(n - 1, lev0, -1):
            lev[k] = lev[k + 1] / ratio
        if lev0 > 1:
            lev[lev0 - 1] = -lev[lev0 + 1]
            for k in range(lev0 - 2, 0, -1):
                lev[k] = lev[k + 1] * ratio
    lev[lev0] = 0.0
    return lev[1:n + 1].copy(), lev0 - 1


def compute_levels(vmin: float, vmax: float, n: int = 11, method: str = "lin", *, abs_values: bool = False,
                   locked: bool = False, log_range: float = 256.0, log_ratio: float | None = None,
                   mode: Mode = "standard") -> Levels:
    """Compute ``n`` colour levels for data in ``[vmin, vmax]``.

    ``locked`` corresponds to v407's fixed min/max (``VectSet``): no range shrink is applied.
    For ``method="log"``, ``log_range`` is the ratio between the largest and smallest |level|
    (v407 ``SnapNLev``); ``log_ratio`` fixes the ratio between successive levels instead.
    """
    n = max(int(n), 2)
    prov = Provenance(mode=mode)
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    if mode == "v407":
        if not locked:
            vmax, vmin = _shrink(vmax, vmin, n)
            prov.add("level_shrink")
        if abs_values:
            vmax, vmin = max(abs(vmax), abs(vmin)), max(0.0, vmin)
        if vmax == vmin:
            vmax = vmin + 1
        if method != "log":
            x = np.arange(n) / (n - 1)
            if method == "lin":
                lev = x * (vmax - vmin) + vmin
            elif method == "quad":
                lev = x**2 * (vmax - vmin) + vmin
            elif method == "dquad":
                lev = x**4 * (vmax - vmin) + vmin
            elif method == "cos":
                lev = 0.5 * (1 - np.cos(np.pi * x)) ** 4 * (vmax - vmin) + vmin
                prov.add("cosine_levels")
            elif method == "fixed":
                idx = [pas_trunc(k / (n - 1) * (len(LEV_ARRAY) - 1)) for k in range(n)]
                lev = np.array([LEV_ARRAY[i] for i in idx]) * vmax
            else:
                raise ValueError(f"unknown level method {method!r}")
            lev = lev.astype(float)
            lev0 = 0
            for k in range(1, n):
                if abs(lev[k]) < abs(lev[lev0]):
                    lev0 = k
            lev[lev0] = 0.0
            prov.add("zero_level", "band_toward_zero")
            return Levels(lev, method, vmin, vmax, abs_values, lev0, None, prov)
        ratio = log_ratio or (calc_one_level_ratio(vmax, vmin, n, log_range) if log_range > 0 else 2.0)
        lev, lev0 = _log_levels_v407(vmax, vmin, n, ratio)
        prov.add("zero_level", "band_toward_zero")
        return Levels(lev, method, vmin, vmax, abs_values, lev0, ratio, prov)

    # --- standard mode ---------------------------------------------------------------------
    if abs_values:
        vmax, vmin = max(abs(vmax), abs(vmin)), 0.0
    if vmax == vmin:
        vmax = vmin + (abs(vmin) or 1.0)
    x = np.linspace(0.0, 1.0, n)
    if method == "lin":
        lev = vmin + x * (vmax - vmin)
    elif method == "quad":
        lev = vmin + x**2 * (vmax - vmin)
    elif method == "dquad":
        lev = vmin + x**4 * (vmax - vmin)
    elif method == "cos":
        lev = vmin + 0.5 * (1 - np.cos(np.pi * x)) * (vmax - vmin)
    elif method == "fixed":
        amax = max(abs(vmax), abs(vmin))
        xs = np.linspace(-1.0, 1.0, n) if vmin < 0 < vmax else (x if vmax > 0 else -x[::-1])
        lev = np.sign(xs) * np.interp(np.abs(xs), np.linspace(0, 1, len(LEV_ARRAY)), LEV_ARRAY) * amax
    elif method == "log":
        lev = _symlog_levels(vmin, vmax, n, log_range, log_ratio)
    else:
        raise ValueError(f"unknown level method {method!r}")
    lev = np.asarray(lev, dtype=float)
    zero_idx = int(np.argmin(np.abs(lev))) if (lev.min() < 0 < lev.max() and method == "log") else None
    return Levels(lev, method, vmin, vmax, abs_values, zero_idx, log_ratio, prov)


def _symlog_levels(vmin: float, vmax: float, n: int, log_range: float, log_ratio: float | None) -> np.ndarray:
    """Levels evenly spaced in a symmetric-log sense, spanning ``log_range`` decades of ratio."""
    amax = max(abs(vmin), abs(vmax))
    if amax == 0:
        return np.linspace(-1, 1, n)
    rng = log_range if log_range > 1 else 256.0
    if log_ratio:
        rng = log_ratio ** max(n // 2, 1)
    lin = amax / rng

    def fwd(v):
        return np.sign(v) * np.log1p(np.abs(v) / lin)

    def inv(u):
        return np.sign(u) * lin * np.expm1(np.abs(u))

    return inv(np.linspace(fwd(vmin), fwd(vmax), n))


def robust_range(a: np.ndarray, percentile: float = 99.5, symmetric: bool = False) -> tuple[float, float]:
    """Range covering the central ``percentile`` % of finite values (outlier-resistant)."""
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(a, 100 - percentile))
    hi = float(np.percentile(a, percentile))
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def level_colors(levels: Levels, palette: list[str], zero_color: str | None = None) -> list[str]:
    """Colour for each level: v407 picks palette index ``(Q_cols·k) div (Q_lev+1)`` (k = 1..Q)."""
    n, ncol = levels.n, len(palette)
    out = []
    for k in range(1, n + 1):
        if levels.provenance.mode == "v407":
            idx = min((ncol * k) // (n + 1), ncol - 1)
        else:
            idx = round((k - 1) / max(n - 1, 1) * (ncol - 1))
        out.append(palette[idx])
    if zero_color is not None and levels.zero_index is not None:
        out[levels.zero_index] = zero_color
    return out
