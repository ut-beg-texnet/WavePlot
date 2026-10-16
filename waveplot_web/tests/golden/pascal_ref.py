"""Loop transliterations of WavePlot v407 numerics (1-based where the Pascal is), used as golden
references for the vectorised ``mode="v407"`` implementations. Line numbers refer to
``Turbo pascal version/``."""

from __future__ import annotations

import math

import numpy as np


def f32(x):
    return float(np.float32(x))


# FOURIER.PAS:89-165 ------------------------------------------------------------------------------
def fft_ref(four: list[complex], invert: bool) -> list[complex]:
    n = len(four)
    a = [complex(c) for c in four]
    n0 = round(math.log(n) / math.log(2))
    for i in range(n):
        b = i
        for n1 in range(n0 // 2):
            n2 = n0 - n1 - 1
            b1 = ((b >> n1) & 1) << n2
            b2 = ((b >> n2) & 1) << n1
            b &= ~(1 << n2)
            b &= ~(1 << n1)
            b |= b1 | b2
        if i < b:
            a[i], a[b] = a[b], a[i]
    nn = 2
    while nn <= n:
        thet = 2 * math.pi / nn
        if invert:
            thet = -thet
        dw = complex(math.cos(thet), math.sin(thet))
        w = complex(1, 0)
        nstep = nn // 2
        for k in range(nn // 2):
            for l in range(n // nn):
                i = l * nn + k
                j = i + nstep
                temp = a[j] * w
                a[j] = a[i] - temp
                a[i] = a[i] + temp
            w = w * dw
        nn *= 2
    if invert:
        a = [c / n for c in a]
    return a


# DSPFORM.PAS:753-782, 790-810, 1084-1098, 992-1003 -----------------------------------------------
def integrate_ref(h: list[float], dt: float) -> list[float]:
    h = list(h)
    prev2 = h[0]
    h[0] = 0.0
    for t in range(1, len(h)):
        prev1 = h[t]
        h[t] = h[t - 1] + prev2 * dt
        prev2 = prev1
    return h


def differentiate_ref(h: list[float], dt: float) -> list[float]:
    h = list(h)
    T = len(h)
    for t in range(T - 1):
        h[t] = (h[t + 1] - h[t]) / dt
    h[T - 1] = h[T - 2]
    return h


def shift_ref(h: list[float], shift: int) -> list[float]:
    h = [0.0] + list(h)   # 1-based
    T = len(h) - 1
    if abs(shift) > T:
        shift = 0
    if shift >= 0:
        for t in range(T, shift, -1):
            h[t] = h[t - shift]
        for t in range(shift, 0, -1):
            h[t] = h[1]
    else:
        shift = -shift
        for t in range(1, T - shift + 1):
            h[t] = h[t + shift]
        for t in range(T - shift + 1, T + 1):
            h[t] = h[T]
    return h[1:]


def decimate_ref(h: list[float], q: int) -> list[float]:
    h = [0.0] + list(h)
    T = len(h) - 1
    T2 = T // q
    for t in range(1, T2 + 1):
        h[t] = h[t * q]
    return h[1:T2 + 1]


def convolve_ref(x: list[float], g: list[float], dt: float) -> list[float]:
    x = [0.0] + list(x)
    g = [0.0] + list(g)
    T = len(x) - 1
    out = [0.0] * (T + 1)
    for t in range(1, T + 1):
        h = 0.0
        for t1 in range(1, T + 1):
            if t > t1:
                h += x[t1] * g[t - t1] * dt
        out[t] = h
    return out[1:]


# DSPFORM.PAS:263-381 + FOURIER.PAS:167-227 ---------------------------------------------------------
def fourier_ref(y: list[float], dt: float, kind: str, filt: str, params: list[float], freqwind: int = 1,
                low_freq=10.0, high_freq=100.0, low_taper=5.0, high_taper=95.0, max_taper=5.0):
    T = len(y)
    pc = len(params)
    P = list(params) + [0.0] * (5 - pc)
    partype = 1 if filt == "pass" else 0
    # SetFiltParams (ParStart=1)
    maxf = 0.5 / dt
    lf, hf = low_freq / 100, high_freq / 100
    if partype == 0:
        if pc >= 1:
            lf = P[0] / 100
        if pc >= 2:
            hf = P[1] / 100
    else:
        if pc >= 1:
            lf = P[0] / maxf
        if pc >= 2:
            hf = P[1] / maxf
    # SetTaperParams (ParStart=3)
    ts1 = math.trunc(0.01 * low_taper * T)
    ts2 = math.trunc(0.01 * high_taper * T)
    tn0 = math.trunc(0.01 * max_taper * T)
    if partype == 0:
        if pc >= 3:
            ts1 = math.trunc(0.01 * P[2] * (T - 1) + 1)
        if pc >= 4:
            ts2 = math.trunc(0.01 * P[3] * (T - 1) + 1)
        if pc >= 5:
            tn0 = math.trunc(0.01 * P[4] * T)
    else:
        if pc >= 3:
            ts1 = math.trunc(P[2] / dt + 1)
        if pc >= 4:
            ts2 = math.trunc(P[3] / dt + 1)
        if pc >= 5:
            tn0 = math.trunc(P[4] / dt)
    N = math.trunc(math.log(T - 1) / math.log(2)) + 1 + freqwind
    N = round(math.exp(N * math.log(2)))
    N = min(N, 4096)
    st = (N - T) // 2
    fin = st + T - 1
    four = [0j] * N
    for i in range(1, T + 1):
        four[st - 1 + i] = complex(y[i - 1], 0)
    tn = min(tn0, ts1)
    ts0 = ts1 - tn
    for i in range(0, ts0):
        four[st + i] = complex(0, four[st + i].imag)
    for i in range(ts0, ts1 + 1):
        tau = (i - ts1) / (tn + 1)
        four[st + i] = four[st + i] * math.exp(-10 * tau * tau)
    tn = tn0
    if tn > fin - st - ts2:
        tn = fin - st - ts2
    tsf = ts2 + tn
    for i in range(ts2, tsf + 1):
        tau = (i - ts2) / (tn + 1)
        four[st + i] = four[st + i] * math.exp(-10 * tau * tau)
    for i in range(tsf + 1, fin - st + 1):
        four[st + i] = 0j
    four = fft_ref(four, False)
    if filt != "none":
        fst, fend = math.trunc(N * lf) // 2, math.trunc(N * hf) // 2
        if filt == "stop":
            for i in range(fst, fend + 1):
                four[i] = 0j
                four[N - 1 - i] = 0j
        else:
            for i in range(0, fst + 1):
                d = math.exp(-(fst - i) * 0.01)
                four[i] *= d
                four[N - 1 - i] *= d
            for i in range(fend, N // 2):
                d = math.exp(-(i - fend) * 0.01)
                four[i] *= d
                four[N - 1 - i] *= d
    if kind == "inverse":
        four = fft_ref(four, True)
        return [c.real for c in four], N, st
    out = []
    for i in range(1, N + 1):
        c = four[i - 1]
        if kind == "amp":
            out.append(abs(c) * dt)
        else:
            z, yy = c.imag, c.real
            if yy == 0:
                ang = math.pi / 2 if z > 0 else -math.pi / 2
            else:
                ang = math.atan(z / yy)
                if yy < 0:
                    ang = ang + math.pi if z > 0 else ang - math.pi
            v = ang * 180 / math.pi
            if v < 0:
                v += 360
            if i > 1:
                if v < out[-1]:
                    v = math.floor((out[-1] - v) / 360 + 1) * 360 + v
                if v - out[-1] > 360:
                    v -= 360
            out.append(v)
    return out, N, st


# DSPFORM.PAS:218-261 ----------------------------------------------------------------------------
def applyq_ref(y: list[float], dt: float, Q: float, C: float, L: float, freqwind: int = 1) -> list[float]:
    T = len(y)
    N = math.trunc(math.log(T - 1) / math.log(2)) + 1 + freqwind
    N = min(round(math.exp(N * math.log(2))), 4096)
    st = (N - T) // 2
    four = [0j] * N
    for i in range(1, T + 1):
        four[st - 1 + i] = complex(y[i - 1], 0)
    four = fft_ref(four, False)
    df = 1 / dt / N
    f = 0.0
    qterm = -(2 * math.pi * L) / (2 * C * Q)
    for i in range(0, N // 2):
        q = math.exp(qterm * f)
        four[i] *= q
        four[N - 1 - i] *= q
        f += df
    four = fft_ref(four, True)
    return [four[st - 1 + i].real for i in range(1, T + 1)]


# DSPFORM.PAS:1208-1295 ----------------------------------------------------------------------------
def tau_p_ref(h: list[float], dt: float, typ: int, ds: float | None, asecs: float, rtyp: int) -> list[float]:
    h = list(h)
    T = len(h)
    h2 = [0.0] * T
    dsmall = 1e-20
    alpha = math.exp(math.log(0.1) / (asecs / dt))
    if typ == 0:
        for t in range(1, T - 1):
            h2[t] = (h[t + 1] - h[t - 1]) / dt * 0.5
        h2[0] = (h[1] - h[0]) / dt
        h2[T - 1] = (h[T - 1] - h[T - 2]) / dt
        nns = min(50, T // 10)
        nns = max(nns, 2)
        for t in range(nns):
            dsmall += h[t] * h[t]
        dsmall /= nns
        dsmall = dsmall * asecs / dt
        dsmall = dsmall * 40 / (100.0 if ds is None else ds) ** 2
        if ds is not None:
            dsmall = ds
    xi = di = 0.0
    out = []
    for t in range(T):
        xi = alpha * xi + h[t] * h[t]
        di = alpha * di + h2[t] * h2[t]
        ti = 2 * math.pi * math.sqrt(xi / (di + dsmall))
        out.append(ti if rtyp == 0 else (math.sqrt(xi) if rtyp == 1 else math.sqrt(di)))
    return out


# DSPFORM.PAS:152-191 (Extract) ----------------------------------------------------------------------
def extract_ref(h: list[float], t0: float, dt: float, params: list[float], low_taper=5.0, high_taper=95.0,
                max_taper=5.0):
    T = len(h)
    pc = len(params)
    P = list(params) + [0.0] * (5 - pc)
    partype = 1 if (pc >= 1 and math.trunc(P[0]) == 1) else 0
    ts1 = math.trunc(0.01 * low_taper * T)
    ts2 = math.trunc(0.01 * high_taper * T)
    tn = math.trunc(0.01 * max_taper * T)
    if partype == 0:
        if pc >= 2:
            ts1 = math.trunc(0.01 * P[1] * (T - 1) + 1)
        if pc >= 3:
            ts2 = math.trunc(0.01 * P[2] * (T - 1) + 1)
        if pc >= 4:
            tn = math.trunc(0.01 * P[3] * T)
    else:
        if pc >= 2:
            ts1 = math.trunc((P[1] - t0) / dt + 1)
        if pc >= 3:
            ts2 = math.trunc((P[2] - t0) / dt + 1)
        if pc >= 4:
            tn = math.trunc(P[3] / dt)
    if ts1 < 1 or ts1 > T:
        ts1 = 1
    if ts2 < ts1 or ts2 > T:
        ts2 = T
    ts0 = max(ts1 - tn, 1)
    tsf = min(ts2 + tn, T)
    nt = math.trunc(P[3]) if pc >= 5 else 1
    if nt < 1 or nt > 100:
        nt = 1
    buf = [0.0] + list(h)
    for tt in range(1, tsf - ts0 + 2):
        if tt % nt == 0:
            tau = 0.0
            t = tt + ts0 - 1
            if t < ts1:
                tau = (ts1 - t) / tn
            if t > ts2:
                tau = (t - ts2) / tn
            buf[tt // nt] = buf[t] * math.exp(-10 * tau * tau)
    tq = (tsf - ts0) // nt + 1
    return buf[1:tq + 1], t0 + (ts0 - 1) * dt, dt * nt
