"""Ground-motion style metrics and response spectra for time series."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate as _integrate


@dataclass
class TraceMetrics:
    peak: float              # max |y|
    peak_time: float
    peak_to_peak: float
    rms: float
    mean: float
    dominant_freq: float | None
    duration_5_95: float | None   # significant duration from cumulative y² (Husid)
    t5: float | None
    t95: float | None
    energy: float            # ∫ y² dt
    cav: float               # ∫ |y| dt (cumulative absolute value)


def trace_metrics(y: np.ndarray, t0: float, dt: float) -> TraceMetrics:
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if not good.any():
        nan = float("nan")
        return TraceMetrics(nan, nan, nan, nan, nan, None, None, None, None, nan, nan)
    yy = np.where(good, y, 0.0)
    k = int(np.argmax(np.abs(yy)))
    energy_cum = _integrate.cumulative_trapezoid(yy**2, dx=dt, initial=0.0)
    total = energy_cum[-1]
    t5 = t95 = dur = None
    if total > 0:
        tt = t0 + dt * np.arange(len(y))
        t5 = float(np.interp(0.05 * total, energy_cum, tt))
        t95 = float(np.interp(0.95 * total, energy_cum, tt))
        dur = t95 - t5
    fdom = None
    if len(y) > 8:
        X = np.abs(np.fft.rfft(yy - yy.mean()))
        f = np.fft.rfftfreq(len(y), dt)
        if X[1:].size:
            fdom = float(f[1 + int(np.argmax(X[1:]))])
    return TraceMetrics(
        peak=float(abs(yy[k])), peak_time=float(t0 + k * dt), peak_to_peak=float(yy.max() - yy.min()),
        rms=float(math.sqrt(np.mean(yy**2))), mean=float(yy.mean()), dominant_freq=fdom, duration_5_95=dur,
        t5=t5, t95=t95, energy=float(total), cav=float(np.trapezoid(np.abs(yy), dx=dt)),
    )


def response_spectrum(acc: np.ndarray, dt: float, periods: np.ndarray, damping: float = 0.05
                      ) -> dict[str, np.ndarray]:
    """Elastic SDOF response spectra (exact piecewise-linear recurrence, Nigam & Jennings 1969).

    Returns peak relative displacement ``sd``, pseudo-velocity ``psv`` and pseudo-acceleration
    ``psa`` for each period (vectorised over periods). ``acc`` is the base acceleration.
    """
    acc = np.asarray(acc, float)
    periods = np.asarray(periods, float)
    T = np.where(periods > 0, periods, np.nan)
    z = float(damping)
    w = 2 * np.pi / T
    sq = math.sqrt(1 - z * z)
    wd = w * sq
    e = np.exp(-z * w * dt)
    s, c = np.sin(wd * dt), np.cos(wd * dt)
    a11 = e * (z / sq * s + c)
    a12 = e / wd * s
    a21 = -w / sq * e * s
    a22 = e * (c - z / sq * s)
    t1 = (2 * z * z - 1) / (w * w * dt)
    t2 = 2 * z / (w**3 * dt)
    b11 = e * ((t1 + z / w) * s / wd + (t2 + 1 / (w * w)) * c) - t2
    b12 = -e * (t1 * s / wd + t2 * c) - 1 / (w * w) + t2
    b21 = e * ((t1 + z / w) * (c - z / sq * s) - (t2 + 1 / (w * w)) * (wd * s + z * w * c)) + 1 / (w * w * dt)
    b22 = -e * (t1 * (c - z / sq * s) - t2 * (wd * s + z * w * c)) - 1 / (w * w * dt)
    u = np.zeros_like(w)
    v = np.zeros_like(w)
    umax = np.zeros_like(w)
    for i in range(len(acc) - 1):
        p0, p1 = -acc[i], -acc[i + 1]
        u, v = (a11 * u + a12 * v + b11 * p0 + b12 * p1, a21 * u + a22 * v + b21 * p0 + b22 * p1)
        np.maximum(umax, np.abs(u), out=umax)
    sd = np.nan_to_num(umax)
    ww = np.nan_to_num(w)
    return {"period": periods, "sd": sd, "psv": sd * ww, "psa": sd * ww * ww}
