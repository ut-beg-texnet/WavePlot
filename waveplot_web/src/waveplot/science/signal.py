"""Time-domain trace operations (integration, differentiation, windows, shifts, decimation, ...).

Functions take NumPy arrays plus the sample interval and return float64 arrays. Where WavePlot
v407 behaved differently, ``mode="v407"`` reproduces it exactly (transliterated from
``DSPFORM.PAS``); ``mode="standard"`` uses conventional, accurate numerics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate as _integrate
from scipy import signal as _signal

from .compat import Mode, pas_pow, pas_trunc


@dataclass(frozen=True)
class FourierSettings:
    """v407 Fourier / taper settings (Settings → Fourier); percentages as in WVPL407.SET."""

    oversize: int = 1          # FreqWind: FFT length = next power of two × 2**oversize
    low_freq: float = 10.0     # default stop/pass band start, % of Nyquist
    high_freq: float = 100.0   # default band end, % of Nyquist
    low_taper: float = 5.0     # start of window, % of samples
    high_taper: float = 95.0   # end of window, % of samples
    max_taper: float = 5.0     # taper length, % of samples


DEFAULT_FOURIER = FourierSettings()


def integrate(y: np.ndarray, dt: float, mode: Mode = "standard") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    if mode == "v407":   # left rectangle rule, y[0] = 0 (DSPFORM.PAS:753-767)
        return np.concatenate([[0.0], np.cumsum(y[:-1]) * dt])
    return _integrate.cumulative_trapezoid(y, dx=dt, initial=0.0)


def differentiate(y: np.ndarray, dt: float, mode: Mode = "standard") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.zeros_like(y)
    if mode == "v407":   # forward difference, last value repeated (DSPFORM.PAS:768-782)
        d = np.diff(y) / dt
        return np.append(d, d[-1])
    return np.gradient(y, dt, edge_order=2 if len(y) > 2 else 1)


def sample_shift(y: np.ndarray, n: int) -> np.ndarray:
    """Shift by ``n`` samples, padding with the edge value (``S``); |n| > len leaves it unchanged."""
    y = np.asarray(y, dtype=float)
    n = int(n)
    if abs(n) > len(y) or n == 0:
        return y.copy()
    if n > 0:
        return np.concatenate([np.full(n, y[0]), y[:-n]])
    n = -n
    return np.concatenate([y[n:], np.full(n, y[-1])])


def power(y: np.ndarray, p: float, mode: Mode = "standard") -> np.ndarray:
    """``E``: x^p. Standard: integer p → true power, otherwise signed power sign(x)·|x|^p."""
    y = np.asarray(y, dtype=float)
    if mode == "v407":
        return pas_pow(y, p)
    if float(p).is_integer():
        return y ** int(p)
    return np.sign(y) * np.abs(y) ** p


def unwrap_steps(y: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """``L``: keep successive increments within [lo, hi] by adding/subtracting (hi − lo)."""
    y = np.asarray(y, dtype=float).copy()
    limit = hi - lo
    if limit <= 0:
        raise ValueError("unwrap: need hi > lo")
    prev = y[0]
    for t in range(1, len(y)):
        v = y[t]
        if v - prev > hi:
            v -= math.ceil((v - prev - hi) / limit) * limit
        if v - prev < lo:
            v += math.ceil((lo - (v - prev)) / limit) * limit
        y[t] = v
        prev = v
    return y


@dataclass(frozen=True)
class WindowResult:
    y: np.ndarray
    t0: float
    dt: float


def _taper_params(n: int, t0: float, dt: float, units: str, t1, t2, taper, s: FourierSettings
                  ) -> tuple[int, int, int]:
    """``SetTaperParams``: (ts1, ts2, tn) as 1-based sample numbers / count (v407 semantics)."""
    ts1 = pas_trunc(0.01 * s.low_taper * n)
    ts2 = pas_trunc(0.01 * s.high_taper * n)
    tn = pas_trunc(0.01 * s.max_taper * n)
    if units == "percent":
        if t1 is not None:
            ts1 = pas_trunc(0.01 * t1 * (n - 1) + 1)
        if t2 is not None:
            ts2 = pas_trunc(0.01 * t2 * (n - 1) + 1)
        if taper is not None:
            tn = pas_trunc(0.01 * taper * n)
    else:
        if t1 is not None:
            ts1 = pas_trunc((t1 - t0) / dt + 1)
        if t2 is not None:
            ts2 = pas_trunc((t2 - t0) / dt + 1)
        if taper is not None:
            tn = pas_trunc(taper / dt)
    return ts1, ts2, max(tn, 0)


def window(y: np.ndarray, t0: float, dt: float, units: str = "percent", t1: float | None = None,
           t2: float | None = None, taper: float | None = None, decimate: int | None = None,
           settings: FourierSettings = DEFAULT_FOURIER, mode: Mode = "standard") -> WindowResult:
    """Extract a part of a trace with Gaussian tapers exp(−10τ²) on both sides (``J`` / ``&W``)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if mode == "v407":
        return _window_v407(y, t0, dt, units, t1, t2, taper, decimate, settings)
    # standard: keep [t1, t2] fully, taper over `taper` outside, then decimate properly
    tt = t0 + dt * np.arange(n)
    dur = tt[-1] - tt[0] if n > 1 else dt
    if units == "percent":
        a = tt[0] + (settings.low_taper if t1 is None else t1) / 100 * dur
        b = tt[0] + (settings.high_taper if t2 is None else t2) / 100 * dur
        w = (settings.max_taper if taper is None else taper) / 100 * dur
    else:
        a = tt[0] if t1 is None else t1
        b = tt[-1] if t2 is None else t2
        w = settings.max_taper / 100 * dur if taper is None else taper
    a, b = max(a, tt[0]), min(b, tt[-1])
    if b < a:
        a, b = tt[0], tt[-1]
    lo, hi = max(a - w, tt[0]), min(b + w, tt[-1])
    keep = (tt >= lo - 1e-12 * abs(dt)) & (tt <= hi + 1e-12 * abs(dt))
    tau = np.zeros(n)
    if w > 0:
        tau = np.where(tt < a, (a - tt) / w, np.where(tt > b, (tt - b) / w, 0.0))
    out = (y * np.exp(-10 * tau**2))[keep]
    t0n = float(tt[keep][0]) if keep.any() else t0
    q = int(decimate or 1)
    if q > 1:
        out = out[::q]
    return WindowResult(out, t0n, dt * q)


def _window_v407(y, t0, dt, units, t1, t2, taper, decimate, s) -> WindowResult:
    n = len(y)
    ts1, ts2, tn = _taper_params(n, t0, dt, units, t1, t2, taper, s)
    if ts1 < 1 or ts1 > n:
        ts1 = 1
    if ts2 < ts1 or ts2 > n:
        ts2 = n
    ts0 = max(ts1 - tn, 1)
    tsf = min(ts2 + tn, n)
    # v407 bug: with 5 parameters the decimation factor is read from the taper (Params[4])
    nt = pas_trunc(taper) if (decimate is not None and taper is not None) else 1
    if nt < 1 or nt > 100:
        nt = 1
    buf = np.concatenate([[0.0], y])          # 1-based buffer
    src = buf.copy()
    for tt in range(1, tsf - ts0 + 2):
        if tt % nt == 0:
            t = tt + ts0 - 1
            tau = 0.0
            if t < ts1:
                tau = (ts1 - t) / tn
            if t > ts2:
                tau = (t - ts2) / tn
            buf[tt // nt] = src[t] * math.exp(-10 * tau * tau)
    nq = (tsf - ts0) // nt + 1
    return WindowResult(buf[1:nq + 1].copy(), t0 + (ts0 - 1) * dt, dt * nt)


def decimate(y: np.ndarray, t0: float, dt: float, q: int, mode: Mode = "standard") -> WindowResult:
    """``&D``: reduce the sampling rate by an integer factor ``q``.

    Standard: anti-alias FIR filter (zero phase), samples 0, q, 2q, … keep the start time.
    v407: no filter; samples q, 2q, … (1-based) with the start time unchanged.
    """
    y = np.asarray(y, dtype=float)
    q = max(int(q), 1)
    if q == 1:
        return WindowResult(y.copy(), t0, dt)
    if mode == "v407":
        m = len(y) // q
        return WindowResult(y[q - 1:q * m:q][:m].copy(), t0, dt * q)
    if len(y) > 3 * 30:
        out = _signal.decimate(y, q, ftype="fir", zero_phase=True)
    else:
        out = y[::q]
    return WindowResult(np.asarray(out, float), t0, dt * q)


def convolve(x: np.ndarray, h: np.ndarray, dt: float, mode: Mode = "standard") -> np.ndarray:
    """``$*``: causal convolution. v407 sums t1 < t (one-sample delay); standard includes t1 = t."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    n = len(x)
    full = np.convolve(x, h[:n])[:n] * dt
    if mode == "v407":
        return np.concatenate([[0.0], full[:-1]])
    return full


def reflect_g(x1: np.ndarray, x2: np.ndarray, p: int, s: int) -> np.ndarray:
    """``$g,P,S``: boundary-reflection sub-function g (recursive; DSPFORM.PAS:955-973)."""
    n = len(x1)
    g = np.zeros(n + 1)
    a, b = np.concatenate([[0.0], x1]), np.concatenate([[0.0], x2])
    acc = 0.0
    for t in range(1, n + 1):
        h = 0.0
        if 1 <= t + p <= n:
            h += b[t + p]
        if 1 <= t + p - s <= n:
            h -= a[t + p - s]
            h += g[t + p - s]
        if 1 <= t - p <= n:
            h += g[t - p]
        if 1 <= t - s <= n:
            h -= g[t - s]
        h -= 0.5 * acc
        g[t] = h
        acc += h
    return g[1:]


def reflect_h(x1: np.ndarray, x2: np.ndarray, p: int, s: int) -> np.ndarray:
    """``$h,P,S``: boundary-reflection sub-function h (DSPFORM.PAS:974-990)."""
    n = len(x1)
    hh = np.zeros(n + 1)
    a, b = np.concatenate([[0.0], x1]), np.concatenate([[0.0], x2])
    for t in range(1, n + 1):
        h = 0.0
        if 1 <= t + p <= n:
            h -= b[t + p]
        h += a[t]
        if 1 <= t + p - s <= n:
            h += hh[t + p - s]
        if 1 <= t - s <= n:
            h += hh[t - s]
        if 1 <= t + p - 2 * s <= n:
            h -= hh[t + p - 2 * s]
        hh[t] = h
    return hh[1:]


def reflect_G(x1: np.ndarray, x2: np.ndarray, p: int) -> np.ndarray:
    """``$G,P``: h = ½(x1[t] + x2[t+P]); out[t] = h[t] + out[t−P] (DSPFORM.PAS:1005-1021)."""
    n = len(x1)
    h = np.zeros(n + 1)
    a, b = np.concatenate([[0.0], x1]), np.concatenate([[0.0], x2])
    for t in range(1, n + 1):
        if t <= n - p:
            h[t] = 0.5 * (a[t] + b[t + p])
    out = a.copy()
    for t in range(1, n + 1):
        out[t] = h[t] if t <= p else h[t] + out[t - p]
    return out[1:]


def envelope(y: np.ndarray) -> np.ndarray:
    """Amplitude envelope |analytic signal| (Hilbert transform)."""
    return np.abs(_signal.hilbert(np.asarray(y, float)))


def detrend(y: np.ndarray, kind: str = "linear") -> np.ndarray:
    return _signal.detrend(np.asarray(y, float), type=kind)


def butter_simple(y: np.ndarray, dt: float, cutoff: float, btype: str, order: int = 4,
                  zero_phase: bool = True) -> np.ndarray:
    nyq = 0.5 / dt
    wn = min(max(cutoff / nyq, 1e-6), 0.999999)
    sos = _signal.butter(order, wn, btype=btype, output="sos")
    y = np.asarray(y, float)
    if zero_phase and len(y) > 3 * (2 * len(sos) + 1):
        return _signal.sosfiltfilt(sos, y)
    return _signal.sosfilt(sos, y)
