"""Operation registry: one declaration per operation drives evaluation, the pipeline editor's
parameter forms, the legacy (v407) mapping and the in-app reference documentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..science import filters as _filt
from ..science import rotation as _rot
from ..science import signal as _sig
from ..science import spectra as _spec
from ..science import tp as _tp
from .trace import Trace


class FormulaError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    name: str
    default: Any = None
    kind: str = "float"          # float | int | choice | bool
    doc: str = ""
    unit: str = ""
    choices: tuple = ()


@dataclass(frozen=True)
class OpSpec:
    name: str
    title: str
    category: str
    arity: int                   # number of trace inputs (0 = reads histories itself)
    impl: Callable
    params: tuple[Param, ...] = ()
    legacy: str = ""             # legacy syntax summary for the docs
    doc: str = ""
    infix: str | None = None     # printed as an infix operator (+ - * / **)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


OPS: dict[str, OpSpec] = {}


def op(name: str, title: str, category: str, arity: int = 1, params: tuple[Param, ...] = (), legacy: str = "",
       doc: str = "", infix: str | None = None):
    def deco(fn):
        OPS[name] = OpSpec(name, title, category, arity, fn, params, legacy, doc, infix)
        return fn
    return deco


def _num(p: dict, name: str, default=None):
    v = p.get(name, default)
    return default if v is None else v


def _int_unit(u: str) -> str:
    return {"m/s": "m", "m/s²": "m/s", "m/s2": "m/s"}.get(u, f"{u}·s" if u else "")


def _diff_unit(u: str) -> str:
    return {"m": "m/s", "m/s": "m/s²"}.get(u, f"{u}/s" if u else "1/s")


# --- unary -------------------------------------------------------------------------------------
@op("reverse", "Reverse in time", "basic", legacy="X")
def _reverse(a, p, ctx):
    return a[0].with_y(a[0].y[::-1].copy())


@op("scale", "Multiply by a constant", "basic", params=(Param("c", 1.0, doc="factor"),), legacy="M,c")
def _scale(a, p, ctx):
    return a[0].with_y(a[0].y * float(_num(p, "c", 1.0)))


@op("offset", "Add a constant", "basic", params=(Param("c", 0.0, doc="value added"),), legacy="#,c")
def _offset(a, p, ctx):
    return a[0].with_y(a[0].y + float(_num(p, "c", 0.0)))


@op("neg", "Negate", "basic", doc="−x")
def _neg(a, p, ctx):
    return a[0].with_y(-a[0].y)


@op("abs", "Absolute value", "basic")
def _abs(a, p, ctx):
    return a[0].with_y(np.abs(a[0].y))


@op("integrate", "Integrate", "calculus", legacy="I",
    doc="Cumulative integral (standard: trapezoid; v407: left rectangle rule, first value 0).")
def _integrate(a, p, ctx):
    t = a[0]
    if ctx.mode == "v407":
        ctx.provenance.add("integrate_left_rect")
    return t.with_y(_sig.integrate(t.y, t.dt, ctx.mode), y_unit=_int_unit(t.y_unit))


@op("differentiate", "Differentiate", "calculus", legacy="D",
    doc="Time derivative (standard: 2nd-order central differences; v407: forward difference).")
def _differentiate(a, p, ctx):
    t = a[0]
    if ctx.mode == "v407":
        ctx.provenance.add("diff_forward")
    return t.with_y(_sig.differentiate(t.y, t.dt, ctx.mode), y_unit=_diff_unit(t.y_unit))


@op("shift_time", "Shift time axis", "axis", params=(Param("tau", 0.0, unit="s", doc="added to the axis"),),
    legacy="s,τ", doc="Relabels the axis only; values are unchanged.")
def _shift_time(a, p, ctx):
    return a[0].replace(t0=a[0].t0 + float(_num(p, "tau", 0.0)))


@op("shift_samples", "Shift samples", "axis", params=(Param("n", 0, "int", doc="samples (positive = delay)"),),
    legacy="S,n", doc="Moves values by n samples, padding with the edge value.")
def _shift_samples(a, p, ctx):
    return a[0].with_y(_sig.sample_shift(a[0].y, int(_num(p, "n", 0))))


@op("power", "Power", "basic", params=(Param("p", 0.5, doc="exponent"),), legacy="E,p",
    doc="Integer exponents give true powers; otherwise sign(x)|x|^p (v407: its own sign rule).")
def _power(a, p, ctx):
    if ctx.mode == "v407":
        ctx.provenance.add("pow_sign")
    return a[0].with_y(_sig.power(a[0].y, float(_num(p, "p", 0.5)), ctx.mode))


@op("window", "Window / extract", "window",
    params=(Param("units", "percent", "choice", choices=("percent", "absolute"),
                  doc="percent of the record or absolute times"),
            Param("t1", None, doc="window start (% or s)"), Param("t2", None, doc="window end (% or s)"),
            Param("taper", None, doc="Gaussian taper length (% or s)"),
            Param("decimate", None, "int", doc="keep every n-th sample")),
    legacy="J,typ,t1,t2,taper,n  or  &W", doc="Keeps [t1, t2] and tapers with exp(−10τ²) outside.")
def _window(a, p, ctx):
    t = a[0]
    r = _sig.window(t.y, t.t0, t.dt, p.get("units") or "percent", p.get("t1"), p.get("t2"), p.get("taper"),
                    p.get("decimate"), ctx.settings, ctx.mode)
    return t.with_y(r.y, t0=r.t0, dt=r.dt)


@op("unwrap", "Unwrap steps", "basic",
    params=(Param("hi", 180.0, doc="max increment"), Param("lo", -180.0, doc="min increment")), legacy="L,hi,lo",
    doc="Adds/subtracts (hi − lo) until successive increments lie within [lo, hi] (phase unwrapping).")
def _unwrap(a, p, ctx):
    return a[0].with_y(_sig.unwrap_steps(a[0].y, float(_num(p, "hi", 180.0)), float(_num(p, "lo", -180.0))))


_SPEC_PARAMS = (
    Param("kind", "amp", "choice", choices=("amp", "phase", "inverse"), doc="output"),
    Param("filter", "none", "choice", choices=("none", "stop", "pass"), doc="FFT band filter"),
    Param("f1", None, doc="band start (% of Nyquist or Hz)"), Param("f2", None, doc="band end (% of Nyquist or Hz)"),
    Param("t1", None, doc="taper window start (% or s)"), Param("t2", None, doc="taper window end (% or s)"),
    Param("taper", None, doc="taper length (% or s)"),
    Param("units", None, "choice", choices=("percent", "absolute"), doc="default: percent; absolute for pass"),
)


@op("spectrum", "Fourier spectrum / FFT filter", "spectral", params=_SPEC_PARAMS,
    legacy="F f P p T t  &F &P &p",
    doc="Amplitude (|X|·dt) or phase spectrum, or the waveform after an FFT stop/pass-band filter.")
def _spectrum(a, p, ctx):
    t = a[0]
    r = _spec.spectrum(t.y, t.t0, t.dt, p.get("kind") or "amp", p.get("filter") or "none", p.get("f1"),
                       p.get("f2"), p.get("t1"), p.get("t2"), p.get("taper"), p.get("units"), ctx.settings,
                       ctx.mode)
    ctx.provenance.add(*r.quirks)
    kind = p.get("kind") or "amp"
    if kind == "inverse":
        return t.with_y(r.y, t0=r.t0, dt=r.dt)
    unit = "deg" if kind == "phase" else (f"{t.y_unit}·s" if t.y_unit else "")
    return t.with_y(r.y, t0=r.t0, dt=r.dt, x_unit="Hz", y_unit=unit,
                    scale_hint="log" if kind == "amp" else None,
                    meta={**t.meta, "full_spectrum": r.full, "spectrum_kind": kind, "time_dt": t.dt})


@op("decimate", "Decimate", "window", params=(Param("q", 2, "int", doc="factor"),), legacy="&D,q",
    doc="Standard: anti-alias filter first. v407: plain sub-sampling (samples q, 2q, …).")
def _decimate(a, p, ctx):
    t = a[0]
    if ctx.mode == "v407":
        ctx.provenance.add("decimate_no_aa")
    r = _sig.decimate(t.y, t.t0, t.dt, int(_num(p, "q", 2)), ctx.mode)
    return t.with_y(r.y, t0=r.t0, dt=r.dt)


@op("bandpass", "Butterworth band-pass", "filter",
    params=(Param("f_lo", 0.0, unit="Hz"), Param("f_hi", 0.0, unit="Hz"),
            Param("sections", 4, "int", doc="number of 4-pole sections (order = 2×sections)"),
            Param("zero_phase", 1, "int", doc="1 = forward + backward (zero phase)")),
    legacy="&f,flo,fhi,ns,zph",
    doc="f_lo = 0 → low-pass, f_hi ≥ Nyquist → high-pass (standard mode).")
def _bandpass(a, p, ctx):
    t = a[0]
    zp = bool(int(_num(p, "zero_phase", 1)))
    if ctx.mode == "v407" and zp:
        ctx.provenance.add("butter_state")
    return t.with_y(_filt.bandpass(t.y, t.dt, float(_num(p, "f_lo", 0.0)), float(_num(p, "f_hi", 0.0)),
                                   int(_num(p, "sections", 4)), zp, ctx.mode))


@op("lowpass", "Butterworth low-pass", "filter",
    params=(Param("f", 1.0, unit="Hz"), Param("order", 4, "int"), Param("zero_phase", 1, "int")))
def _lowpass(a, p, ctx):
    t = a[0]
    return t.with_y(_sig.butter_simple(t.y, t.dt, float(_num(p, "f", 1.0)), "lowpass", int(_num(p, "order", 4)),
                                       bool(int(_num(p, "zero_phase", 1)))))


@op("highpass", "Butterworth high-pass", "filter",
    params=(Param("f", 1.0, unit="Hz"), Param("order", 4, "int"), Param("zero_phase", 1, "int")))
def _highpass(a, p, ctx):
    t = a[0]
    return t.with_y(_sig.butter_simple(t.y, t.dt, float(_num(p, "f", 1.0)), "highpass", int(_num(p, "order", 4)),
                                       bool(int(_num(p, "zero_phase", 1)))))


@op("attenuate", "Constant-Q attenuation", "filter",
    params=(Param("Q", 1e6), Param("c", 6000.0, unit="m/s", doc="wave speed"),
            Param("L", 1.0, unit="m", doc="propagation distance")),
    legacy="&q,Q,C,L", doc="Amplitude-only (acausal) exp(−π f L / (c Q)).")
def _attenuate(a, p, ctx):
    t = a[0]
    if ctx.mode == "v407":
        ctx.provenance.add("mirror_bin", "fft_sign")
    return t.with_y(_spec.constant_q(t.y, t.dt, float(_num(p, "Q", 1e6)), float(_num(p, "c", 6000.0)),
                                     float(_num(p, "L", 1.0)), ctx.settings, ctx.mode))


@op("tp_window", "τp pre-window", "seismology",
    params=(Param("start", 10.0, unit="s"), Param("window", 10.0, unit="s"), Param("mirror", 0, "int")),
    legacy="&x,start,window,typ", doc="Exponentially damped window ending at 'start' (truncated or mirrored).")
def _tp_window(a, p, ctx):
    t = a[0]
    y = _tp.tp_window(t.y, t.dt, float(_num(p, "start", 10.0)), float(_num(p, "window", 10.0)),
                      bool(int(_num(p, "mirror", 0))))
    return t.with_y(y)


_TP_PARAMS = (Param("kind", 0, "int", doc="0 velocity (Nakamura), 1 displacement (Kanamori), 2 acceleration"),
              Param("ds", None, doc="noise level ε (default: estimated)"),
              Param("window", 10.0, unit="s", doc="averaging window"),
              Param("output", 0, "int", doc="0 τp, 1 √X, 2 √D"))


@op("tau_p", "Predominant period τp", "seismology", params=_TP_PARAMS, legacy="&t,typ,Ds,window,out")
def _tau_p(a, p, ctx):
    t = a[0]
    y = _tp.predominant_period(t.y, t.dt, int(_num(p, "kind", 0)), p.get("ds"), float(_num(p, "window", 10.0)),
                               int(_num(p, "output", 0)))
    return t.with_y(y, y_unit="s" if int(_num(p, "output", 0)) == 0 else t.y_unit)


@op("tau_p_damped", "Predominant period τp (damping-corrected)", "seismology", params=_TP_PARAMS,
    legacy="&T,typ,Ds,window,out")
def _tau_p_damped(a, p, ctx):
    t = a[0]
    y = _tp.predominant_period(t.y, t.dt, int(_num(p, "kind", 0)), p.get("ds"), float(_num(p, "window", 10.0)),
                               int(_num(p, "output", 0)), experimental=True)
    return t.with_y(y, y_unit="s" if int(_num(p, "output", 0)) == 0 else t.y_unit)


@op("envelope", "Envelope (Hilbert)", "basic", doc="|analytic signal|")
def _envelope(a, p, ctx):
    return a[0].with_y(_sig.envelope(a[0].y))


@op("demean", "Remove mean", "basic")
def _demean(a, p, ctx):
    return a[0].with_y(a[0].y - np.nanmean(a[0].y))


@op("detrend", "Remove linear trend", "basic")
def _detrend(a, p, ctx):
    return a[0].with_y(_sig.detrend(a[0].y))


@op("normalize", "Normalise to peak 1", "basic")
def _normalize(a, p, ctx):
    m = np.nanmax(np.abs(a[0].y)) or 1.0
    return a[0].with_y(a[0].y / m, y_unit="")


# --- binary --------------------------------------------------------------------------------------
@op("add", "Add", "arithmetic", 2, legacy="+", infix="+")
def _add(a, p, ctx):
    return _binary(a, ctx, np.add)


@op("sub", "Subtract", "arithmetic", 2, legacy="-", infix="-")
def _sub(a, p, ctx):
    return _binary(a, ctx, np.subtract)


@op("mul", "Multiply", "arithmetic", 2, legacy="*", infix="*")
def _mul(a, p, ctx):
    return _binary(a, ctx, np.multiply, unit="product")


@op("div", "Divide", "arithmetic", 2, legacy="/", infix="/",
    doc="Where the divisor is 0 the result keeps the dividend (v407) / is NaN (standard).")
def _div(a, p, ctx):
    def f(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            if ctx.mode == "v407":
                return np.where(y != 0, x / np.where(y == 0, 1, y), x)
            return np.where(y != 0, x / np.where(y == 0, 1, y), np.nan)
    return _binary(a, ctx, f, unit="ratio")


@op("pow", "Power (trace or constant exponent)", "arithmetic", 2, infix="**")
def _pow(a, p, ctx):
    with np.errstate(invalid="ignore"):
        return _binary(a, ctx, np.power, unit="none")


@op("rotate", "Rotate two components", "components", 2, params=(Param("theta", 0.0, unit="deg"),),
    legacy="C,θ", doc="x·cosθ + y·sinθ")
def _rotate(a, p, ctx):
    th = float(_num(p, "theta", 0.0))
    return _binary(a, ctx, lambda x, y: _rot.rotate2(x, y, th))


@op("hypot", "Magnitude of two components", "components", 2, legacy="R", doc="√(x² + y²)")
def _hypot(a, p, ctx):
    return _binary(a, ctx, _rot.hypot)


@op("angle", "Direction of two components", "components", 2, legacy="r", doc="atan2(y, x) in degrees")
def _angle(a, p, ctx):
    return _binary(a, ctx, lambda x, y: _rot.angle_deg(x, y, ctx.mode), unit="deg")


@op("convolve", "Convolve", "signal", 2, legacy="$*", doc="Causal convolution ∫ x(τ) h(t−τ) dτ")
def _convolve(a, p, ctx):
    dt = a[0].dt
    return _binary(a, ctx, lambda x, y: _sig.convolve(x, y, dt, ctx.mode), unit="product")


@op("reflect_g", "Boundary reflection g", "signal", 2, params=(Param("p", 0, "int"), Param("s", 0, "int")),
    legacy="$g,P,S")
def _reflect_g(a, p, ctx):
    return _binary(a, ctx, lambda x, y: _sig.reflect_g(x, y, int(_num(p, "p", 0)), int(_num(p, "s", 0))))


@op("reflect_h", "Boundary reflection h", "signal", 2, params=(Param("p", 0, "int"), Param("s", 0, "int")),
    legacy="$h,P,S")
def _reflect_h(a, p, ctx):
    return _binary(a, ctx, lambda x, y: _sig.reflect_h(x, y, int(_num(p, "p", 0)), int(_num(p, "s", 0))))


@op("reflect_G", "Boundary reflection G", "signal", 2, params=(Param("p", 0, "int"),), legacy="$G,P")
def _reflect_G(a, p, ctx):
    return _binary(a, ctx, lambda x, y: _sig.reflect_G(x, y, int(_num(p, "p", 0))))


@op("from_spectrum", "Waveform from amplitude & phase", "spectral", 2, legacy="G(amp)(phase)",
    doc="Inverse transform of an amplitude spectrum (1st input) and a phase spectrum in degrees (2nd).")
def _from_spectrum(a, p, ctx):
    amp, ph = a
    af = amp.meta.get("full_spectrum")
    pf = ph.meta.get("full_spectrum")
    if ctx.mode == "v407":
        # v407 keeps all N bins in the buffer after 'F'/'P'; use them when available
        A = af if (af is not None and not np.iscomplexobj(af)) else amp.y
        P = pf if (pf is not None and not np.iscomplexobj(pf)) else ph.y
        n = 2 * (amp.n - 1)
        A = np.resize(np.asarray(A, float), n)
        P = np.resize(np.asarray(P, float), n)
        y, dt = _spec.inverse_from_spectrum(A, P, amp.dt, "v407")
    else:
        y, dt = _spec.inverse_from_spectrum(amp.y, ph.y, amp.dt, "standard")
    return Trace(y, 0.0, dt, "s", "", amp.label)


# --- group operations (read several histories) ---------------------------------------------------
def _hist_triplet(p, ctx) -> tuple[int, list[Trace]]:
    h = p.get("h")
    h = int(h) if h else ctx.self_ref
    if not h:
        raise FormulaError("no history given (parameter h) and no trace slot history")
    return h, [ctx.load(h + k) for k in range(3)]


@op("vector_magnitude", "Vector magnitude of 3 histories", "components", 0,
    params=(Param("h", None, "int", doc="first of three consecutive histories (x, y, z)"),), legacy="&V,h")
def _vector_magnitude(a, p, ctx):
    h, (x, y, z) = _hist_triplet(p, ctx)
    n = min(x.n, y.n, z.n)
    return x.with_y(_rot.vector_magnitude(x.y[:n], y.y[:n], z.y[:n]), label=f"|h{h}..h{h + 2}|")


@op("attenuated_magnitude", "Q-attenuated vector magnitude", "components", 0,
    params=(Param("Q", 1e6), Param("c", 6000.0, unit="m/s"), Param("L", 1.0, unit="m"),
            Param("h", None, "int", doc="first of three consecutive histories")), legacy="&Q,Q,C,L,h")
def _attenuated_magnitude(a, p, ctx):
    _h, comps = _hist_triplet(p, ctx)
    att = [_spec.constant_q(c.y, c.dt, float(_num(p, "Q", 1e6)), float(_num(p, "c", 6000.0)),
                            float(_num(p, "L", 1.0)), ctx.settings, ctx.mode) for c in comps]
    n = min(len(v) for v in att)
    return comps[0].with_y(_rot.vector_magnitude(*(v[:n] for v in att)))


@op("rotate_to_source", "Rotate 3 components towards a source", "components", 0,
    params=(Param("component", 0, "int", doc="0 radial, 1 SV-like, 2 transverse"),
            Param("h", None, "int", doc="first of three consecutive histories (x, y, z)"),
            Param("i", 0.0, doc="source grid i"), Param("j", 0.0, doc="source grid j"), Param("k", 0.0,
                                                                                            doc="source grid k")),
    legacy="&R,comp,h,i,j,k")
def _rotate_to_source(a, p, ctx):
    h, (x, y, z) = _hist_triplet(p, ctx)
    if ctx.mode == "v407":
        ctx.provenance.add("index_rotation")
    pos = ctx.position(h)
    gam, thet = _rot.source_angles(pos, (float(_num(p, "i", 0)), float(_num(p, "j", 0)), float(_num(p, "k", 0))),
                                   ctx.spacing, ctx.mode)
    n = min(x.n, y.n, z.n)
    return x.with_y(_rot.rotate_to_source(x.y[:n], y.y[:n], z.y[:n], gam, thet, int(_num(p, "component", 0))))


# --- helpers ---------------------------------------------------------------------------------------
def _combine_units(u1: str, u2: str, how: str | None) -> str:
    if how == "product":
        return f"{u1}·{u2}".strip("·")
    if how == "ratio":
        return "" if u1 == u2 else f"{u1}/{u2}"
    if how == "none":
        return ""
    if how == "deg":
        return "deg"
    return u1 or u2


def _binary(a: list[Trace], ctx, f, unit: str | None = None) -> Trace:
    x, y = a
    scalar_x, scalar_y = x.meta.get("scalar"), y.meta.get("scalar")
    if scalar_x or scalar_y:
        base = y if scalar_x else x
        return base.with_y(f(x.y if not scalar_x else x.y[0], y.y if not scalar_y else y.y[0]))
    if ctx.mode == "v407":
        # index-wise; the result takes the right operand's axis and length (global Hmap)
        ctx.provenance.add("binary_right_time")
        n = y.n
        xv = np.zeros(n)
        xv[:min(n, x.n)] = x.y[:n]
        return y.with_y(f(xv, y.y), y_unit=_combine_units(x.y_unit, y.y_unit, unit), label="")
    if x.n == y.n and np.isclose(x.t0, y.t0) and np.isclose(x.dt, y.dt):
        yv = y.y
    else:
        yv = np.interp(x.x, y.x, y.y, left=np.nan, right=np.nan)
        ctx.warn(f"inputs have different sampling; '{y.label or 'right'}' interpolated onto "
                 f"'{x.label or 'left'}' (NaN outside the overlap)")
    return x.with_y(f(x.y, yv), y_unit=_combine_units(x.y_unit, y.y_unit, unit), label="")


CATEGORIES = ["basic", "calculus", "axis", "window", "filter", "spectral", "seismology", "components", "arithmetic",
              "signal"]


def ops_by_category(arity: int | None = None) -> dict[str, list[OpSpec]]:
    out: dict[str, list[OpSpec]] = {}
    for spec in OPS.values():
        if arity is not None and spec.arity != arity:
            continue
        out.setdefault(spec.category, []).append(spec)
    return {c: out[c] for c in CATEGORIES if c in out}


_ = field
