"""Butterworth band-pass filtering (``&f``; FILTERS.PAS ``bp_bworth``).

v407 builds ``ns`` cascaded 4th-order band-pass sections with the bilinear transform (the poles
of a 2·ns-order low-pass prototype). Zero phase = filter forward, then backward *continuing
from the forward pass's internal state* (not ``filtfilt``). Standard mode designs the same
filter order with SciPy and uses ``sosfiltfilt`` (or ``sosfilt`` when causal).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as _signal

from .compat import Mode


def bworth_sections_v407(dt: float, flo: float, fhi: float, ns: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-section ``(b, a)`` coefficients exactly as ``bp_bworth`` designs them."""
    w1 = math.sin(flo * math.pi * dt) / math.cos(flo * math.pi * dt)
    w2 = math.sin(fhi * math.pi * dt) / math.cos(fhi * math.pi * dt)
    wc = w2 - w1
    q = wc * wc + 2.0 * w1 * w2
    s = w1 * w1 * w2 * w2
    out = []
    for k in range(1, ns + 1):
        c3 = (2.0 * (k + ns) - 1.0) * math.pi / (4 * ns)
        p = -2.0 * wc * math.cos(c3)
        r = p * w1 * w2
        x = 1.0 + p + q + r + s
        a = wc * wc / x
        b = (-4.0 - 2.0 * p + 2.0 * r + 4.0 * s) / x
        c = (6.0 - 2.0 * q + 6.0 * s) / x
        d = (-4.0 + 2.0 * p - 2.0 * r + 4.0 * s) / x
        e = (1.0 - p + q - r + s) / x
        out.append((np.array([a, 0.0, -2.0 * a, 0.0, a]), np.array([1.0, b, c, d, e])))
    return out


def bandpass(y: np.ndarray, dt: float, flo: float, fhi: float, sections: int = 4, zero_phase: bool = True,
             mode: Mode = "standard") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    sections = max(int(sections), 1)
    if mode == "v407":
        return _bandpass_v407(y, dt, flo, fhi, sections, zero_phase)
    nyq = 0.5 / dt
    order = 2 * sections
    if flo <= 0 and fhi <= 0:
        return np.zeros_like(y)
    if flo <= 0:
        sos = _signal.butter(order, min(fhi / nyq, 0.999999), btype="lowpass", output="sos")
    elif fhi <= 0 or fhi >= nyq:
        sos = _signal.butter(order, flo / nyq, btype="highpass", output="sos")
    else:
        sos = _signal.butter(order, [flo / nyq, min(fhi / nyq, 0.999999)], btype="bandpass", output="sos")
    if zero_phase:
        padlen = min(3 * (2 * len(sos) + 1), len(y) - 1)
        return _signal.sosfiltfilt(sos, y, padlen=max(padlen, 0))
    return _signal.sosfilt(sos, y)


def _bandpass_v407(y: np.ndarray, dt: float, flo: float, fhi: float, ns: int, zero_phase: bool) -> np.ndarray:
    secs = bworth_sections_v407(dt, flo, fhi, ns)
    x = y.astype(np.float32).astype(float)        # v407 stores the trace as Single
    histories = []
    sig = x
    for b, a in secs:                              # forward pass, zero initial state
        out = _signal.lfilter(b, a, sig)
        histories.append((sig, out))
        sig = out
    fwd = sig.astype(np.float32).astype(float)
    if not zero_phase:
        return fwd
    # backward pass: each section continues from its forward-pass memory (last 4 inputs/outputs)
    sig = fwd[::-1]
    for (b, a), (xin, yout) in zip(secs, histories, strict=True):
        xi = _last4(xin)
        yi = _last4(yout)
        zi = _signal.lfiltic(b, a, yi, xi)
        sig, _ = _signal.lfilter(b, a, sig, zi=zi)
    return sig[::-1].astype(np.float32).astype(float)


def _last4(v: np.ndarray) -> np.ndarray:
    """Most recent first, zero-padded to 4 (``lfiltic`` convention)."""
    tail = v[-4:][::-1]
    return np.pad(tail, (0, 4 - len(tail)))


def bandpass_reference_v407(y: np.ndarray, dt: float, flo: float, fhi: float, ns: int, zph: int) -> np.ndarray:
    """Loop transliteration of ``bp_bworth`` (used as the golden reference in tests)."""
    tr = np.asarray(y, dtype=np.float32).copy()
    secs = bworth_sections_v407(dt, flo, fhi, ns)
    A = [0.0] + [s[0][0] for s in secs]
    B = [0.0] + [s[1][1] for s in secs]
    C = [0.0] + [s[1][2] for s in secs]
    D = [0.0] + [s[1][3] for s in secs]
    E = [0.0] + [s[1][4] for s in secs]
    f = [[0.0] * 6 for _ in range(ns + 2)]

    def run(indices):
        for m in indices:
            f[1][5] = float(tr[m])
            for n in range(1, ns + 1):
                temp = A[n] * (f[n][5] - 2.0 * f[n][3] + f[n][1])
                temp = temp - B[n] * f[n + 1][4] - C[n] * f[n + 1][3]
                f[n + 1][5] = temp - D[n] * f[n + 1][2] - E[n] * f[n + 1][1]
            for n in range(1, ns + 2):
                for mm in range(1, 5):
                    f[n][mm] = f[n][mm + 1]
            tr[m] = f[ns + 1][5]

    run(range(len(tr)))
    if zph == 1:
        run(range(len(tr) - 1, -1, -1))
    return tr.astype(float)
