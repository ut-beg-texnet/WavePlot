from __future__ import annotations

import math

import numpy as np
import pytest

from waveplot.science import metrics, signal, spectra, stress
from waveplot.science.trajectories import trajectory


def test_principal_2d_cases():
    p = stress.principal_2d(np.array([-10.0]), np.array([-2.0]), np.array([0.0]))
    assert p.sigma_min[0] == -10 and p.sigma_max[0] == -2
    assert abs(math.cos(p.angle_min[0])) == pytest.approx(1.0)       # compression along ax1
    p = stress.principal_2d(np.array([0.0]), np.array([0.0]), np.array([5.0]))   # pure shear
    assert p.sigma_max[0] == pytest.approx(5) and p.sigma_min[0] == pytest.approx(-5)
    assert abs(p.angle_min[0]) == pytest.approx(math.pi / 4)
    v = stress.principal_2d(np.array([-10.0]), np.array([-2.0]), np.array([0.0]), mode="v407")
    assert v.sigma_min[0] == -10
    # the direction really is an eigenvector of σmin (rotated tensor check)
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b, c = rng.normal(size=3)
        for mode in ("standard", "v407"):
            q = stress.principal_2d(np.array([a]), np.array([b]), np.array([c]), mode=mode)
            d = np.array([math.cos(q.angle_min[0]), math.sin(q.angle_min[0])])
            T = np.array([[a, c], [c, b]])
            np.testing.assert_allclose(T @ d, q.sigma_min[0] * d, atol=1e-9)


def test_rotation_consistency_and_v407_sign():
    s11, s22, s12 = -8.0, -3.0, 2.0
    # standard rotation: the rotated tensor keeps its invariants
    for th in (0, 17, 45, 90, 133):
        n11, n22, t12 = stress.rotated(s11, s22, s12, th)
        assert n11 + n22 == pytest.approx(s11 + s22)
        assert n11 * n22 - t12**2 == pytest.approx(s11 * s22 - s12**2)
    # v407 differs only through the sign of the ½(s11−s22)sin2θ term
    a = stress.ess(s11, s22, s12, 30, 30)
    b = stress.ess(s11, s22, s12, 30, 30, mode="v407")
    t_std = -0.5 * (s11 - s22) * math.sin(math.radians(60)) + s12 * math.cos(math.radians(60))
    t_old = +0.5 * (s11 - s22) * math.sin(math.radians(60)) + s12 * math.cos(math.radians(60))
    assert float(a) - float(b) == pytest.approx(abs(t_std) - abs(t_old))
    assert stress.ess(s11, s22, s12, 0, 30) == stress.ess(s11, s22, s12, 0, 30, mode="v407")


def test_ess_optimal_and_principal_3d():
    best, _ang = stress.ess_optimal(np.array([-10.0]), np.array([-2.0]), np.array([0.0]), 30)
    grid = max(float(stress.ess(-10.0, -2.0, 0.0, th, 30)) for th in range(180))
    assert best[0] == pytest.approx(grid)
    p = stress.principal_3d(1.0, 2.0, 3.0, 0.5, 0.2, 0.1)
    assert p.values.sum() == pytest.approx(6.0)
    assert p.values[0] >= p.values[1] >= p.values[2]
    assert stress.von_mises(1, 1, 1, 0, 0, 0) == pytest.approx(0.0)
    m = stress.mohr(-8.0, -3.0, 2.0)
    (c, r), = m.circles
    assert c == pytest.approx(-5.5) and r == pytest.approx(math.sqrt(4 + 6.25))


def test_standard_signal_ops():
    dt = 0.001
    t = np.arange(0, 2, dt)
    w = 2 * np.pi * 3
    y = np.cos(w * t)
    np.testing.assert_allclose(signal.integrate(y, dt), np.sin(w * t) / w, atol=5e-6)   # O(dt²) trapezoid error
    np.testing.assert_allclose(signal.differentiate(np.sin(w * t), dt)[5:-5], (w * np.cos(w * t))[5:-5], rtol=1e-4,
                               atol=1e-3)
    r = signal.decimate(np.sin(w * t), 0.0, dt, 4)
    assert r.dt == pytest.approx(4 * dt) and len(r.y) == pytest.approx(len(t) / 4, abs=1)
    sp = spectra.spectrum(np.sin(w * t), 0.0, dt, "amp")
    f = sp.t0 + sp.dt * np.arange(len(sp.y))
    assert f[np.argmax(sp.y)] == pytest.approx(3.0, abs=0.1)
    inv = spectra.spectrum(np.sin(w * t), 0.0, dt, "inverse", filt="none", t1=0, t2=100, taper=0)
    np.testing.assert_allclose(inv.y, np.sin(w * t), atol=1e-6)


def test_response_spectrum_matches_lsim():
    from scipy import signal as ss

    dt = 0.005
    t = np.arange(0, 10, dt)
    acc = np.sin(2 * np.pi * 1.3 * t) * np.exp(-0.2 * t)
    rs = metrics.response_spectrum(acc, dt, np.array([0.4, 0.77, 1.5]), 0.05)
    for T, sd in zip(rs["period"], rs["sd"], strict=True):
        wn = 2 * np.pi / T
        _, u, _ = ss.lsim(ss.lti([-1], [1, 2 * 0.05 * wn, wn * wn]), acc, t)
        assert sd == pytest.approx(np.abs(u).max(), rel=1e-6)


def test_metrics_and_trajectory():
    dt = 0.01
    y = np.zeros(1000)
    y[500] = 1.0
    m = metrics.trace_metrics(y, 0.0, dt)
    assert m.peak == 1.0 and m.peak_time == pytest.approx(5.0)
    v = np.ones(10)
    path = trajectory(v, None, v * 2, dt=0.1, kind="disp")
    assert path[-1, 0] == pytest.approx(0.9) and path[-1, 2] == pytest.approx(1.8)
    acc_std = trajectory(np.arange(5.0), None, None, dt=1.0, kind="acc")
    acc_old = trajectory(np.arange(5.0), None, None, dt=1.0, kind="acc", mode="v407")
    assert acc_std[2, 0] == pytest.approx(1.0) and acc_old[2, 0] == pytest.approx(-1.0)   # v407 sign bug


def test_contours_engines_agree_roughly():
    from waveplot.science.contours import contour_lines

    x = np.linspace(-1, 1, 41)
    y = np.linspace(-1, 1, 31)
    X, Y = np.meshgrid(x, y)
    z = X**2 + Y**2
    for engine in ("contourpy", "v407"):
        ((_lev, xs, ys),) = contour_lines(z, x, y, [0.25], engine)
        r = np.hypot(xs[np.isfinite(xs)], ys[np.isfinite(ys)])
        assert r.size > 20 and np.allclose(r, 0.5, atol=0.02)


def test_ess_optimal_closed_form_matches_search():
    rng = np.random.default_rng(7)
    s11, s22, s12 = (rng.normal(size=300) * 5 - (k == 1) * 3 for k in range(3))
    best, normal = stress.ess_optimal(s11, s22, s12, 30.0, 0.5)
    grid = np.max([stress.ess(s11, s22, s12, th, 30.0, 0.5) for th in np.arange(0, 180, 0.05)], axis=0)
    np.testing.assert_allclose(best, grid, atol=1e-3 * np.abs(grid).max())
    at = np.array([float(stress.ess(s11[k], s22[k], s12[k], normal[k] - 90, 30.0, 0.5)) for k in range(300)])
    np.testing.assert_allclose(at, best, atol=2e-3 * np.abs(grid).max())
