"""Turbo Pascal / WavePlot v407 numeric primitives, used by ``mode="v407"`` code paths.

Every function documents the Pascal behaviour it reproduces. Standard-mode code should not use
these except where the behaviour is identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Mode = Literal["standard", "v407"]

#: Human-readable descriptions of every v407 quirk reproduced in compatibility mode.
QUIRKS: dict[str, str] = {
    "ess_shear_sign": "ESS shear term uses +½(s11−s22)·sin2θ (shear and normal stress on different planes)",
    "fft_sign": "FFT uses e^{+i} (phase sign opposite to the usual convention)",
    "arctan0": "atan2(0,0) = −90° (empty spectral bins show 270° phase)",
    "phase_monotone": "phase spectrum forced to increase monotonically",
    "fft_cap": "FFT length capped at 4096",
    "taper_offset": "spectral taper indices offset by one sample",
    "mirror_bin": "FFT filters modify bin N−1−i instead of the true mirror N−i",
    "butter_state": "zero-phase Butterworth carries filter state into the backward pass",
    "integrate_left_rect": "integration with the left rectangle rule",
    "diff_forward": "differentiation with a forward difference (half-sample shift)",
    "decimate_no_aa": "decimation without anti-alias filter, keeping samples q, 2q, …",
    "pow_sign": "signed power keeps the sign only for odd round(p) or odd round(1/p)",
    "accel_sign": "acceleration trajectories have inverted sign ((v_prev − v)/dt)",
    "level_shrink": "colour-level range shrunk by (Q−0.5)/Q",
    "cosine_levels": "cosine level distribution 0.5·(1−cos πx)⁴ (overshoots the range)",
    "zero_level": "level nearest zero forced to 0 and drawn as background",
    "band_toward_zero": "values take the colour of the nearest level towards zero",
    "dump_sampling": "dump glyph grid picked with ROUND((p−0.49)·ratio)",
    "index_rotation": "rotation towards a source uses grid indices instead of metres",
    "binary_right_time": "binary operations take the right operand's time base, index-wise",
    "hist_truncate": "histories longer than 4096 samples truncated",
    "v407_contours": "contours from the 4-triangle centre-point scheme",
}


@dataclass
class Provenance:
    """Records how a result was computed (shown as a badge in the UI)."""

    mode: Mode = "standard"
    quirks: list[str] = field(default_factory=list)

    def add(self, *names: str) -> None:
        for n in names:
            if n not in self.quirks:
                self.quirks.append(n)

    def describe(self) -> list[str]:
        return [QUIRKS.get(q, q) for q in self.quirks]


def pas_round(x: float | np.ndarray) -> float | np.ndarray:
    """Turbo Pascal ``Round``: half away from zero (Python's ``round`` is half-to-even)."""
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.floor(np.abs(x) + 0.5)
    return math.copysign(math.floor(abs(x) + 0.5), x)


def pas_trunc(x: float) -> int:
    """Turbo Pascal ``Trunc``: towards zero."""
    return math.trunc(x)


def pas_div(a: int, b: int) -> int:
    """Turbo Pascal integer ``DIV``: truncates towards zero (unlike Python ``//``)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def arctan0(z: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """``Arctan0(z, y)`` (FUNCT.PAS:113-130): atan2(z, y) except (0, 0) → −π/2 and (y<0, z=0) → −π."""
    z_arr, y_arr = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    out = np.arctan2(z_arr, y_arr)
    out = np.where((z_arr == 0) & (y_arr == 0), -np.pi / 2, out)
    out = np.where((z_arr == 0) & (y_arr < 0), -np.pi, out)
    return float(out) if out.ndim == 0 else out


def pas_pow(a: np.ndarray, n: float) -> np.ndarray:
    """``POW`` (FUNCT.PAS:393-419): |a|^n, keeping the sign of ``a`` when ``round(n)`` equals
    ``trunc(n)`` and is odd, or else when ``round(1/n)`` equals ``trunc(1/n)`` and is odd."""
    a = np.asarray(a, dtype=float)
    out = np.where(a == 0, 0.0, np.abs(a) ** n)
    keep = False
    if pas_round(n) == pas_trunc(n):
        keep = int(pas_round(n)) % 2 == 1
    elif n != 0 and pas_round(1 / n) == pas_trunc(1 / n):
        keep = int(pas_round(1 / n)) % 2 == 1
    return np.where(keep & (a < 0), -out, out)


def round_places_up(v: float, r: int) -> float:
    """``RoundPlaces`` (FUNCT.PAS:335-352): rounds |v| *up* to ~r significant figures (axis maxima)."""
    if v == 0:
        return 0.0
    d = 10.0 ** (pas_trunc(math.log10(abs(v))) - r)
    return math.copysign(pas_trunc(abs(v) / d + 0.99) * d, v)


def fft_v407(x: np.ndarray) -> np.ndarray:
    """Forward FFT with the v407 sign convention: X[k] = Σ x[n]·e^{+2πikn/N}, unnormalised."""
    return np.conj(np.fft.fft(np.conj(np.asarray(x, dtype=complex))))


def ifft_v407(X: np.ndarray) -> np.ndarray:
    """Inverse of :func:`fft_v407` (divides by N, e^{−2πikn/N})."""
    return np.conj(np.fft.ifft(np.conj(np.asarray(X, dtype=complex))))


def next_pow2_bits(n: int) -> int:
    """Exact ``ceil(log2(n))`` for n ≥ 1 (v407 used a floating-point log)."""
    return max(int(n - 1).bit_length(), 0)
