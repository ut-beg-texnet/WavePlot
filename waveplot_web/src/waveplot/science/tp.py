"""Predominant period τp (Nakamura / Kanamori style; DSPFORM.PAS ``&t``, ``&T``, ``&x``).

    X_t = α X_{t−1} + x_t²,   D_t = α D_{t−1} + ẋ_t²,   τp = 2π √(X / (D + ε)),   α = 0.1^(dt/window)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as _signal


def _central_diff(y: np.ndarray, dt: float) -> np.ndarray:
    d = np.empty_like(y)
    d[1:-1] = (y[2:] - y[:-2]) / dt * 0.5
    d[0] = (y[1] - y[0]) / dt
    d[-1] = (y[-1] - y[-2]) / dt
    return d


def predominant_period(y: np.ndarray, dt: float, kind: int = 0, ds: float | None = None, window: float = 10.0,
                       output: int = 0, experimental: bool = False) -> np.ndarray:
    """τp time series. ``kind``: 0 velocity (Nakamura), 1 displacement (Kanamori), 2 acceleration.

    ``ds`` replaces the noise term ε when given (v407: "if Ds parameter is given, use it directly").
    ``output``: 0 τp, 1 √X, 2 √D. ``experimental`` (``&T``) uses (ẋ − a·x)² with a = ½ ln(0.1)/window.
    """
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n < 3:
        return np.zeros_like(y)
    window = window if window > 0 else 10.0
    alpha = math.exp(math.log(0.1) / (window / dt))
    dsmall = 1e-20
    if kind == 1:
        alpha = 1.0
        y2 = y.copy()
        x = np.empty_like(y)
        x[0] = 0.5 * (y2[1] + y2[0]) * dt
        for t in range(1, n - 1):
            x[t] = x[t - 1] + 0.5 * (y2[t + 1] + y2[t - 1]) * dt
        x[-1] = x[-2] + 0.5 * (y2[-1] + y2[-2]) * dt
        xd = y2
        y = x
    elif kind == 2:
        y1 = _central_diff(y, dt)
        xd = _central_diff(y1, dt)
        y = y1
    else:
        xd = _central_diff(y, dt)
        nns = min(50, n // 10)
        nns = max(nns, 2)
        dsmall = (dsmall + float(np.sum(y[:nns] ** 2))) / nns
        dsmall = dsmall * window / dt
        dsmall = dsmall * 40 / ((ds if ds else 100.0) ** 2)
        if ds is not None and not experimental:
            dsmall = ds
    if experimental:
        a = 0.5 * math.log(0.1) / window
        dterm = (xd - a * y) ** 2
    else:
        dterm = xd**2
    X = _signal.lfilter([1.0], [1.0, -alpha], y**2)
    D = _signal.lfilter([1.0], [1.0, -alpha], dterm)
    if output == 1:
        return np.sqrt(X)
    if output == 2:
        return np.sqrt(D)
    return 2.0 * math.pi * np.sqrt(X / (D + dsmall))


def tp_window(y: np.ndarray, dt: float, start: float = 10.0, window: float = 10.0, mirror: bool = False,
              max_len: int = 4096) -> np.ndarray:
    """``&x``: exponentially damped window ending at ``start`` (truncated, or mirrored to even length)."""
    y = np.asarray(y, dtype=float).copy()
    st = start if start > 0 else 10.0
    st_i = int(st / dt) + 1
    st_i = max(st_i, 20)
    st_i = min(st_i, max_len // 2 - 1, len(y))
    w = window if window > 0 else 10.0
    alpha = math.exp(math.log(0.1) / (w / dt))
    damp = alpha ** np.arange(st_i)              # 1 at st_i, decaying backwards
    seg = y[:st_i] * damp[::-1]
    if not mirror:
        return seg
    return np.concatenate([seg, seg[-2::-1]])
