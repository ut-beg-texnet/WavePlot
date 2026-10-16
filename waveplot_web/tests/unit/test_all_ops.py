"""Every registered operation evaluates in both numerics modes; targeted checks of the rest."""

from __future__ import annotations

import math

import numpy as np
import pytest

from waveplot.formula.evaluate import EvalContext, evaluate
from waveplot.formula.nodes import Call, Hist, Num
from waveplot.formula.ops import OPS, FormulaError
from waveplot.formula.trace import Trace
from waveplot.science import rotation, signal, tp
from waveplot.science.dumpfields import FIELDS, compute_field

DT = 0.01
T = np.arange(400) * DT


def _hist(n: int) -> Trace:
    f = 2.0 + n
    return Trace(np.sin(2 * np.pi * f * T) * np.exp(-((T - 2) / 0.8) ** 2) + 0.01 * n, 0.0, DT, "s", "m/s",
                 f"h{n}")


def _ctx(mode: str) -> EvalContext:
    return EvalContext(load=_hist, self_ref=1, mode=mode, spacing=(10.0, 10.0, 10.0),
                       position=lambda h: (h * 10, 20, 30))


EXAMPLES: dict[str, dict] = {
    "scale": {"c": 2.0}, "offset": {"c": 1.0}, "shift_time": {"tau": 0.5}, "shift_samples": {"n": 3},
    "power": {"p": 2.0}, "window": {"units": "absolute", "t1": 1.0, "t2": 3.0, "taper": 0.2, "decimate": 2},
    "unwrap": {"hi": 0.5, "lo": -0.5}, "decimate": {"q": 3}, "bandpass": {"f_lo": 1.0, "f_hi": 8.0},
    "lowpass": {"f": 5.0}, "highpass": {"f": 1.0}, "attenuate": {"Q": 20.0, "c": 3000.0, "L": 500.0},
    "tp_window": {"start": 2.0, "window": 1.0, "mirror": 1}, "tau_p": {"kind": 0, "window": 1.0},
    "tau_p_damped": {"kind": 1, "window": 1.0}, "rotate": {"theta": 30.0}, "reflect_g": {"p": 2, "s": 1},
    "reflect_h": {"p": 2, "s": 1}, "reflect_G": {"p": 2}, "vector_magnitude": {"h": 1},
    "attenuated_magnitude": {"h": 1, "Q": 50.0}, "rotate_to_source": {"h": 1, "component": 1, "i": 0, "j": 0, "k": 0},
}


@pytest.mark.parametrize("mode", ["standard", "v407"])
@pytest.mark.parametrize("name", sorted(OPS))
def test_every_operation(name, mode):
    spec = OPS[name]
    params = tuple(EXAMPLES.get(name, {}).items())
    if name == "unsupported":
        with pytest.raises(FormulaError):
            evaluate(Call(name, (Hist(1), Hist(2)), (("name", "V"),)), _ctx(mode))
        return
    if name == "spectrum":
        for kind in ("amp", "phase", "inverse"):
            for filt in ("none", "stop", "pass"):
                tr = evaluate(Call(name, (Hist(1),), (("kind", kind), ("filter", filt))), _ctx(mode))
                assert np.isfinite(tr.y).sum() > 0
        return
    if name == "from_spectrum":
        amp = Call("spectrum", (Hist(1),), (("kind", "amp"),))
        ph = Call("spectrum", (Hist(1),), (("kind", "phase"),))
        tr = evaluate(Call(name, (amp, ph)), _ctx(mode))
        assert tr.n > 0 and tr.x_unit == "s"
        return
    args = tuple(Hist(k + 1) for k in range(spec.arity))
    tr = evaluate(Call(name, args, params), _ctx(mode))
    assert tr.n > 0 and np.isfinite(tr.y).any(), name


def test_scalar_arithmetic_and_errors():
    ctx = _ctx("standard")
    tr = evaluate(Call("mul", (Hist(1), Num(2.0))), ctx)
    np.testing.assert_allclose(tr.y, 2 * _hist(1).y)
    with pytest.raises(FormulaError):
        evaluate(Call("integrate", (Num(1.0),)), ctx)
    with pytest.raises(FormulaError):
        evaluate(Call("nope", (Hist(1),)), ctx)


def test_rotation_functions():
    x, y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    np.testing.assert_allclose(rotation.rotate2(x, y, 90), [0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation.hypot(x, y), [1, 1])
    np.testing.assert_allclose(rotation.angle_deg(x, y), [0, 90])
    assert rotation.angle_deg(np.array([0.0]), np.array([0.0]), "v407")[0] == pytest.approx(-90)
    g, t = rotation.source_angles((10, 0, 0), (0, 0, 0), (2.0, 1.0, 1.0))
    assert g == pytest.approx(0.0) and t == pytest.approx(0.0)
    g, t = rotation.source_angles((0, 5, 0), (0, 0, 0), mode="v407")
    assert t == pytest.approx(math.pi / 2)
    vx, vy, vz = np.array([1.0]), np.array([2.0]), np.array([3.0])
    radial = rotation.rotate_to_source(vx, vy, vz, 0.0, 0.0, 0)
    sv = rotation.rotate_to_source(vx, vy, vz, 0.0, 0.0, 1)
    tr = rotation.rotate_to_source(vx, vy, vz, 0.0, 0.0, 2)
    assert (radial[0], sv[0], tr[0]) == (1.0, 2.0, 3.0)


def test_signal_details():
    y = np.array([0.0, 170.0, -170.0, 175.0])
    np.testing.assert_allclose(signal.unwrap_steps(y, 180, -180), [0, 170, 190, 175])
    with pytest.raises(ValueError):
        signal.unwrap_steps(y, 1, 1)
    np.testing.assert_allclose(signal.power(np.array([-8.0, 4.0]), 1 / 3), [-2.0, 4 ** (1 / 3)])
    np.testing.assert_allclose(signal.power(np.array([-2.0]), 2), [4.0])
    w = signal.window(np.ones(101), 0.0, 0.01, "percent", 20, 80, 10)
    assert w.y.max() == pytest.approx(1.0) and w.y[0] < 0.01 and w.t0 == pytest.approx(0.1)
    c = signal.convolve(np.array([1.0, 0, 0]), np.array([1.0, 2, 3]), 1.0)
    np.testing.assert_allclose(c, [1, 2, 3])
    assert signal.envelope(np.sin(np.linspace(0, 20 * np.pi, 2000)))[500:1500].mean() == pytest.approx(1.0, rel=0.02)
    np.testing.assert_allclose(signal.detrend(np.arange(10.0)), 0, atol=1e-12)


def test_tp_variants():
    dt = 0.01
    t = np.arange(0, 20, dt)
    y = np.sin(2 * np.pi * 0.5 * t)
    tau = tp.predominant_period(y, dt, 0, ds=1e-12, window=3.0)
    assert tau[-200:].mean() == pytest.approx(2.0, rel=0.05)
    for kind in (1, 2):
        assert np.isfinite(tp.predominant_period(y, dt, kind, window=3.0)).all()
    assert len(tp.tp_window(y, dt, 5.0, 2.0)) == 501
    assert len(tp.tp_window(y, dt, 5.0, 2.0, mirror=True)) == 1001


def test_all_dump_fields():
    rng = np.random.default_rng(3)
    comps = {k: rng.normal(size=(5, 4)) * (1e5 if k.startswith("s") else 1e-3)
             for k in ("v1", "v2", "v3", "s11", "s22", "s33", "s12", "s23", "s13")}
    ip = {"u": comps["v1"], "v": comps["v2"], "s11": comps["s11"], "s22": comps["s22"], "s12": comps["s12"]}
    for key in FIELDS:
        z = compute_field(key, comps, ip, 20.0, 30.0, 0.0, "standard")
        assert z.shape == (5, 4) and np.isfinite(z).all(), key
    with pytest.raises(ValueError):
        compute_field("vm", {k: v for k, v in comps.items() if k != "s33"}, ip)
