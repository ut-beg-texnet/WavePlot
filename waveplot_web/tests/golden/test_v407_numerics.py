"""v407-mode numerics vs loop transliterations of the Pascal (tests/golden/pascal_ref.py)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.golden import pascal_ref as ref
from waveplot.science import filters, signal, spectra, tp
from waveplot.science.compat import fft_v407, ifft_v407

RNG = np.random.default_rng(42)


def _sig(n=300, dt=0.01):
    t = np.arange(n) * dt
    return (np.sin(2 * np.pi * 3 * t) * np.exp(-((t - 1.2) / 0.4) ** 2) + 0.05 * RNG.normal(size=n)), dt


def close(a, b, rtol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.nanmax(np.abs(b)), 1e-300)
    np.testing.assert_allclose(a, b, rtol=0, atol=rtol * scale)


@pytest.mark.parametrize("n", [8, 64, 256])
def test_fft_sign_convention(n):
    x = RNG.normal(size=n) + 1j * RNG.normal(size=n)
    close(fft_v407(x), ref.fft_ref(list(x), False), 1e-10)
    close(ifft_v407(x), ref.fft_ref(list(x), True), 1e-10)


def test_integrate_differentiate_shift_decimate():
    y, dt = _sig(101)
    close(signal.integrate(y, dt, "v407"), ref.integrate_ref(list(y), dt), 1e-12)
    close(signal.differentiate(y, dt, "v407"), ref.differentiate_ref(list(y), dt), 1e-12)
    for s in (0, 5, -7, 200):
        close(signal.sample_shift(y, s), ref.shift_ref(list(y), s), 0)
    for q in (2, 3, 7):
        r = signal.decimate(y, 0.0, dt, q, "v407")
        close(r.y, ref.decimate_ref(list(y), q), 0)
        assert r.dt == pytest.approx(dt * q) and r.t0 == 0.0


def test_hand_cases():
    np.testing.assert_allclose(signal.integrate([1, 2, 3], 1.0, "v407"), [0, 1, 3])
    np.testing.assert_allclose(signal.differentiate([1, 2, 3], 1.0, "v407"), [1, 1, 1])
    r = signal.decimate(np.array([1.0, 2, 3, 4]), 0.0, 1.0, 2, "v407")
    np.testing.assert_allclose(r.y, [2, 4])
    np.testing.assert_allclose(signal.integrate([1, 2, 3], 1.0), [0, 1.5, 4])   # trapezoid


@pytest.mark.parametrize("params", [[], [0, 10, 60, 5], [1, 0.5, 2.0, 0.2], [0, 20, 80, 3, 2], [0, 5]])
def test_extract(params):
    y, dt = _sig(300)
    units = "absolute" if (params and int(params[0]) == 1) else "percent"
    r = signal.window(y, 0.0, dt, units, *(params[1:4] + [None] * (3 - len(params[1:4]))),
                      decimate=params[4] if len(params) >= 5 else None, mode="v407")
    yy, t0, dtn = ref.extract_ref(list(y), 0.0, dt, params)
    close(r.y, yy, 1e-12)
    assert r.t0 == pytest.approx(t0) and r.dt == pytest.approx(dtn)


@pytest.mark.parametrize("kind", ["amp", "phase", "inverse"])
@pytest.mark.parametrize("filt,params", [("none", []), ("stop", [5, 40]), ("stop", [10, 100, 10, 90, 4]),
                                          ("pass", [2.0, 15.0]), ("none", [0, 0, 20, 70, 10])])
def test_fourier(kind, filt, params):
    y, dt = _sig(300)
    units = "absolute" if filt == "pass" else "percent"
    p = params + [None] * (5 - len(params))
    r = spectra.spectrum(y, 0.0, dt, kind, filt, *p, units=units, mode="v407")
    vals, N, st = ref.fourier_ref(list(y), dt, kind, filt, params)
    if kind == "inverse":
        close(r.y, vals, 1e-9)
        assert r.t0 == pytest.approx(-st * dt)
    else:
        close(r.full, vals, 1e-9)
        assert len(r.y) == N // 2 + 1 and r.dt == pytest.approx(1 / (dt * N))


def test_fourier_empty_bins_phase_270():
    y = np.zeros(64)
    r = spectra.spectrum(y, 0.0, 0.01, "phase", mode="v407")
    assert np.all(np.isclose(r.full % 360, 270))


def test_applyq():
    y, dt = _sig(200)
    close(spectra.constant_q(y, dt, 30.0, 3000.0, 500.0, mode="v407"), ref.applyq_ref(list(y), dt, 30.0, 3000.0,
                                                                                       500.0), 1e-9)


@pytest.mark.parametrize("ns,zph", [(1, 1), (4, 1), (2, 0), (6, 1)])
def test_butterworth_state_carry(ns, zph):
    y, dt = _sig(400)
    got = filters.bandpass(y, dt, 1.0, 8.0, ns, bool(zph), mode="v407")
    want = filters.bandpass_reference_v407(y, dt, 1.0, 8.0, ns, zph)
    close(got, want, 1e-5)


def test_butterworth_standard_matches_scipy_response():
    from scipy import signal as ss

    dt = 0.001
    t = np.arange(4000) * dt
    for f, expect in ((10.0, 1.0), (1.0, 0.0), (200.0, 0.0)):
        y = np.sin(2 * np.pi * f * t)
        out = filters.bandpass(y, dt, 5.0, 20.0, 2, True)
        mid = slice(1000, 3000)
        ratio = np.abs(out[mid]).max() / np.abs(y[mid]).max()
        assert ratio == pytest.approx(expect, abs=0.05)
    _ = ss


@pytest.mark.parametrize("ds,rtyp", [(None, 0), (3.0, 0), (None, 1), (None, 2)])
def test_tau_p(ds, rtyp):
    y, dt = _sig(300)
    close(tp.predominant_period(y, dt, 0, ds, 0.5, rtyp), ref.tau_p_ref(list(y), dt, 0, ds, 0.5, rtyp), 1e-9)


def test_convolution_delay():
    x = RNG.normal(size=50)
    g = RNG.normal(size=50)
    close(signal.convolve(x, g, 0.1, "v407"), ref.convolve_ref(list(x), list(g), 0.1), 1e-12)
