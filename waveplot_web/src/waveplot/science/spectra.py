"""Spectral operations: amplitude/phase spectra, FFT band filters, constant-Q attenuation,
inverse transforms, spectrograms.

v407 (``DSPFORM.PAS`` ``Fourier``/``ApplyQ``/``InvFourier``, ``FOURIER.PAS`` ``F_Filter``):
FFT with e^{+i} sign, length 2^(trunc(log2(n−1))+1+oversize) capped at 4096, data centred in
zero padding, Gaussian end tapers with a one-sample index offset, filters applied to bin i and
N−1−i, phase forced to increase monotonically, amplitude = |X|·dt.

Standard: the same windowing/padding concept with conventional numerics (e^{−i}, true mirror
bins via ``rfft``, no length cap, exact taper indices, ``np.unwrap`` phase).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as _signal

from .compat import Mode, arctan0, fft_v407, ifft_v407, next_pow2_bits, pas_trunc
from .signal import DEFAULT_FOURIER, FourierSettings, _taper_params

F_MAXEL = 4096


@dataclass
class SpectrumResult:
    y: np.ndarray            # values shown (N/2+1 bins for spectra, N samples for inverse)
    t0: float                # start of x axis (0 Hz for spectra)
    dt: float                # x step (Hz for spectra, s for inverse)
    x_unit: str              # "Hz" or "s"
    full: np.ndarray | None = None   # all N bins (needed by the inverse-from-spectrum op)
    quirks: list[str] = field(default_factory=list)


def fft_length(n: int, oversize: int, mode: Mode) -> int:
    if mode == "v407":
        bits = pas_trunc(math.log(max(n - 1, 1)) / math.log(2)) + 1 + oversize
        return min(round(math.exp(bits * math.log(2))), F_MAXEL)
    return 2 ** (next_pow2_bits(n) + oversize)


def _filter_params(dt: float, units: str, f1, f2, s: FourierSettings) -> tuple[float, float]:
    """``SetFiltParams``: band edges as fractions of Nyquist."""
    maxf = 0.5 / dt
    a, b = s.low_freq / 100, s.high_freq / 100
    if units == "percent":
        if f1 is not None:
            a = f1 / 100
        if f2 is not None:
            b = f2 / 100
    else:
        if f1 is not None:
            a = f1 / maxf
        if f2 is not None:
            b = f2 / maxf
    return a, b


def f_filter_v407(X: np.ndarray, filt: str, n: int, f_st: int, f_end: int) -> np.ndarray:
    """``F_Filter``: 'stop' (StepFilt) zeroes bins; 'pass' (SlopeFilt2) rolls off 0.01 Np/bin outside."""
    X = X.copy()
    f_st //= 2
    f_end //= 2
    n2 = n // 2
    if filt == "stop":
        for i in range(f_st, f_end + 1):
            if 0 <= i < n:
                X[i] = 0
                X[n - 1 - i] = 0
    elif filt == "pass":
        for i in range(0, f_st + 1):
            d = math.exp(-(f_st - i) * 0.01)
            X[i] *= d
            X[n - 1 - i] *= d
        for i in range(f_end, n2):
            d = math.exp(-(i - f_end) * 0.01)
            X[i] *= d
            X[n - 1 - i] *= d
    return X


def _taper_v407(four: np.ndarray, st: int, n: int, ts1: int, ts2: int, tn0: int) -> None:
    """In-place v407 tapers on the padded buffer (indices Four[st+i], i = 0-based offsets)."""
    fin_rel = n - 1                      # fin - st
    tn = min(tn0, ts1)
    ts0 = ts1 - tn
    for i in range(0, max(ts0, 0)):
        four[st + i] = 0
    for i in range(ts0, ts1 + 1):
        if 0 <= st + i < len(four):
            four[st + i] *= math.exp(-10 * ((i - ts1) / (tn + 1)) ** 2)
    tn = min(tn0, max(fin_rel - ts2, 0))
    tsf = ts2 + tn
    for i in range(ts2, tsf + 1):
        if 0 <= st + i < len(four):
            four[st + i] *= math.exp(-10 * ((i - ts2) / (tn + 1)) ** 2)
    for i in range(tsf + 1, fin_rel + 1):
        if 0 <= st + i < len(four):
            four[st + i] = 0


def spectrum(y: np.ndarray, t0: float, dt: float, kind: str = "amp", filt: str = "none", f1=None, f2=None,
             t1=None, t2=None, taper=None, units: str | None = None, settings: FourierSettings = DEFAULT_FOURIER,
             mode: Mode = "standard") -> SpectrumResult:
    """Amplitude / phase spectrum or inverse transform of a (filtered) spectrum.

    ``kind``: "amp" | "phase" | "inverse". ``filt``: "none" | "stop" | "pass". ``units``:
    "percent" (band in % of Nyquist, window in % of samples) or "absolute" (Hz and seconds);
    default "percent" except for pass-band filtering (v407 ``&F/&P/&p``).
    """
    y = np.asarray(y, dtype=float)
    units = units or ("absolute" if filt == "pass" else "percent")
    quirks: list[str] = []
    n = len(y)
    if mode == "v407" and n > F_MAXEL:
        y = y[:F_MAXEL]
        n = F_MAXEL
        quirks.append("fft_cap")
    N = fft_length(n, settings.oversize, mode)
    lo, hi = _filter_params(dt, units, f1, f2, settings)
    ts1, ts2, tn0 = _taper_params(n, t0, dt, units, t1, t2, taper, settings)
    st = (N - n) // 2

    if mode == "v407":
        quirks += ["fft_sign", "taper_offset", "mirror_bin"] + (["fft_cap"] if N == F_MAXEL else [])
        four = np.zeros(N, dtype=complex)
        four[st:st + n] = y
        _taper_v407(four, st, n, ts1, ts2, tn0)
        X = fft_v407(four)
        if filt != "none":
            X = f_filter_v407(X, filt, N, pas_trunc(N * lo), pas_trunc(N * hi))
        if kind == "inverse":
            x = ifft_v407(X).real
            return SpectrumResult(x, t0 - st * dt, dt, "s", None, quirks)
        if kind == "amp":
            full = np.abs(X) * dt
        else:
            quirks += ["arctan0", "phase_monotone"]
            ph = np.degrees(arctan0(X.imag, X.real))
            ph = np.where(ph < 0, ph + 360, ph)
            for i in range(1, N):
                if ph[i] < ph[i - 1]:
                    ph[i] = math.floor((ph[i - 1] - ph[i]) / 360 + 1) * 360 + ph[i]
                if ph[i] - ph[i - 1] > 360:
                    ph[i] -= 360
            full = ph
        return SpectrumResult(full[:N // 2 + 1].copy(), 0.0, 1.0 / (dt * N), "Hz", full, quirks)

    # --- standard ---------------------------------------------------------------------------
    buf = np.zeros(N)
    buf[st:st + n] = y
    w = np.ones(n)
    a = max(ts1 - 1, 0)                  # convert 1-based to 0-based sample positions
    b = min(max(ts2 - 1, 0), n - 1)
    k = np.arange(n)
    tn = max(tn0, 0)
    if tn > 0:
        w = np.where(k < a - tn, 0.0, w)
        w = np.where((k >= a - tn) & (k < a), np.exp(-10 * ((k - a) / tn) ** 2), w)
        w = np.where((k > b) & (k <= b + tn), np.exp(-10 * ((k - b) / tn) ** 2), w)
        w = np.where(k > b + tn, 0.0, w)
    else:
        w = np.where((k < a) | (k > b), 0.0, w)
    buf[st:st + n] *= w
    X = np.fft.rfft(buf)
    freqs = np.fft.rfftfreq(N, dt)
    nyq = 0.5 / dt
    if filt == "stop":
        X = np.where((freqs >= lo * nyq) & (freqs <= hi * nyq), 0, X)
    elif filt == "pass":
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        flo, fhi = lo * nyq, hi * nyq
        damp = np.ones_like(freqs)
        damp = np.where(freqs < flo, np.exp(-(flo - freqs) / df * 0.01), damp)
        damp = np.where(freqs > fhi, np.exp(-(freqs - fhi) / df * 0.01), damp)
        X = X * damp
    if kind == "inverse":   # filtered waveform on the original time base
        x = np.fft.irfft(X, N)[st:st + n]
        return SpectrumResult(x, t0, dt, "s", None, quirks)
    if kind == "amp":
        vals = np.abs(X) * dt
    else:
        ph = np.angle(X)
        ph[np.abs(X) == 0] = np.nan
        good = np.isfinite(ph)
        out = np.full_like(ph, np.nan)
        out[good] = np.unwrap(ph[good])
        vals = np.degrees(out)
    return SpectrumResult(vals, 0.0, float(freqs[1] - freqs[0]) if N > 1 else 1.0, "Hz", X, quirks)


def inverse_from_spectrum(amp_full: np.ndarray, phase_full: np.ndarray, df: float, mode: Mode = "standard"
                          ) -> tuple[np.ndarray, float]:
    """``G(amp)(phase)``: time signal from amplitude and phase (degrees). Returns (y, dt)."""
    if mode == "v407":
        N = len(amp_full)
        X = amp_full * np.exp(1j * np.radians(phase_full))
        x = ifft_v407(X).real
        return x, 1.0 / (df * N)
    # standard: one-sided spectra (rfft layout); amplitude was |X|·dt → divide back
    N = 2 * (len(amp_full) - 1)
    dt = 1.0 / (df * N)
    X = (amp_full / dt) * np.exp(1j * np.radians(np.nan_to_num(phase_full)))
    return np.fft.irfft(X, N), dt


def constant_q(y: np.ndarray, dt: float, Q: float, c: float, L: float, settings: FourierSettings = DEFAULT_FOURIER,
               mode: Mode = "standard") -> np.ndarray:
    """Amplitude-only constant-Q attenuation exp(−π f L / (c Q)) (``ApplyQ``; acausal)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    N = max(fft_length(n, settings.oversize, mode), n)
    st = (N - n) // 2
    qterm = -(2 * math.pi * L) / (2 * c * Q)
    if mode == "v407":
        four = np.zeros(N, dtype=complex)
        four[st:st + n] = y
        X = fft_v407(four)
        df = 1.0 / dt / N
        for i in range(0, N // 2):
            f = i * df
            q = math.exp(qterm * f)
            X[i] *= q
            X[N - 1 - i] *= q
        return ifft_v407(X).real[st:st + n].copy()
    buf = np.zeros(N)
    buf[st:st + n] = y
    X = np.fft.rfft(buf)
    f = np.fft.rfftfreq(N, dt)
    return np.fft.irfft(X * np.exp(qterm * f), N)[st:st + n]


def amplitude_spectrum_simple(y: np.ndarray, dt: float, taper_frac: float = 0.05, oversize: int = 1,
                              detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Convenience one-sided Fourier amplitude spectrum |X(f)|·dt with a Tukey taper (Spectra tab)."""
    y = np.asarray(y, float)
    if detrend:
        y = y - y.mean()
    n = len(y)
    w = _signal.windows.tukey(n, 2 * taper_frac) if n > 2 else np.ones(n)
    N = 2 ** (next_pow2_bits(n) + oversize)
    X = np.fft.rfft(y * w, N)
    return np.fft.rfftfreq(N, dt), np.abs(X) * dt


def psd_welch(y: np.ndarray, dt: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    nper = nperseg or min(len(y), max(64, 2 ** (next_pow2_bits(len(y)) - 3)))
    return _signal.welch(y, fs=1.0 / dt, nperseg=nper, detrend="constant")


def spectrogram(y: np.ndarray, dt: float, nperseg: int | None = None, overlap: float = 0.875
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT magnitude in dB: returns (freqs, times, S[f, t])."""
    y = np.asarray(y, float)
    nper = nperseg or max(16, min(256, 2 ** max(next_pow2_bits(len(y)) - 4, 4)))
    nper = min(nper, len(y))
    f, t, Z = _signal.stft(y, fs=1.0 / dt, nperseg=nper, noverlap=int(nper * overlap), boundary=None)
    S = 20 * np.log10(np.abs(Z) + 1e-30)
    return f, t, S
