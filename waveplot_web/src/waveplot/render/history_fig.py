"""Plotly figures for histories: stacked/overlaid traces, spectra, spectrograms, record sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..formula.trace import Trace
from .theme import PALETTE, si_format, template

SCALES = {"lin": "Linear", "abs": "|x|", "log": "log |x|", "loglog": "log–log"}


@dataclass
class TraceView:
    trace: Trace
    label: str
    scale: str = "lin"        # lin | abs | log | loglog
    color: str | None = None
    note: str = ""


def _xy(tv: TraceView, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    x, y = tv.trace.x, tv.trace.y
    if tv.scale in ("abs", "log", "loglog"):
        y = np.abs(y)
    if tv.scale in ("log", "loglog"):
        y = np.where(y > 0, y, np.nan)
    if tv.scale == "loglog":
        x = np.where(x > 0, x, np.nan)
    if len(y) > max_points:   # min/max decimation keeps peaks visible
        k = int(np.ceil(len(y) / (max_points / 2)))
        m = len(y) // k
        yy = y[: m * k].reshape(m, k)
        xx = x[: m * k].reshape(m, k)
        imax = np.nanargmax(np.nan_to_num(yy, nan=-np.inf), axis=1)
        imin = np.nanargmin(np.nan_to_num(yy, nan=np.inf), axis=1)
        idx = np.sort(np.stack([imin, imax], axis=1), axis=1)
        rows = np.repeat(np.arange(m), 2)
        x, y = xx[rows, idx.ravel()], yy[rows, idx.ravel()]
    return x.astype(np.float64), y.astype(np.float64)


def _xtitle(unit: str) -> str:
    return {"s": "time (s)", "m": "distance (m)", "Hz": "frequency (Hz)"}.get(unit, unit)


def traces_figure(views: list[TraceView], *, layout: str = "stacked", vlink: str = "own", vfix: float | None = None,
                  hlink: str = "common", window: tuple[float, float] | None = None, scheme: str = "light",
                  uirevision: str = "hist", markers: bool = False) -> go.Figure:
    if not views:
        from .theme import empty_figure

        return empty_figure("Add histories from the left panel (or click receivers on the Snapshots page)",
                            scheme)
    units = {v.trace.x_unit for v in views}
    shared_x = hlink != "own" and len(units) == 1
    n = len(views) if layout == "stacked" else 1
    fig = make_subplots(rows=n, cols=1, shared_xaxes=shared_x, vertical_spacing=min(0.04, 0.4 / max(n, 1)))
    vmax_all = max((np.nanmax(np.abs(v.trace.y)) for v in views if v.trace.n and np.isfinite(v.trace.y).any()),
                   default=1.0)
    for k, tv in enumerate(views):
        row = k + 1 if layout == "stacked" else 1
        col = tv.color or PALETTE[k % len(PALETTE)]
        x, y = _xy(tv)
        unit = tv.trace.y_unit
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines+markers" if markers else "lines", name=tv.label,
            line=dict(color=col, width=1.3), marker=dict(size=3),
            hovertemplate=f"<b>{tv.label}</b><br>%{{x:.5g}} {tv.trace.x_unit}<br>%{{y:.5g}} {unit}<extra></extra>"),
            row=row, col=1)
        ya = "yaxis" if row == 1 else f"yaxis{row}"
        xa = "xaxis" if row == 1 else f"xaxis{row}"
        yopts: dict = {"exponentformat": "e", "showexponent": "all"}
        if tv.scale in ("log", "loglog"):
            yopts["type"] = "log"
        if tv.scale == "loglog" or (tv.trace.scale_hint == "log" and tv.scale == "loglog"):
            fig.update_layout({xa: {"type": "log"}})
        if layout == "stacked":
            yopts["title"] = {"text": unit or "", "font": {"size": 10}}
            if vlink in ("linked", "fixed"):
                m = (vfix if (vlink == "fixed" and vfix) else vmax_all) * 1.05
                if tv.scale == "lin":
                    yopts["range"] = [-m, m]
                elif tv.scale == "abs":
                    yopts["range"] = [0, m]
            fig.add_annotation(text=f"<b>[{k + 1}]</b> {tv.label}" + (f"  <i>{tv.note}</i>" if tv.note else ""),
                               xref=f"{xa.replace('axis', '')} domain", yref=f"{ya.replace('axis', '')} domain",
                               x=0.005, y=0.98, xanchor="left", yanchor="top", showarrow=False,
                               font=dict(size=11, color=col), bgcolor="rgba(127,127,127,0.08)")
        fig.update_layout({ya: yopts})
        if not shared_x or row == n:
            fig.update_layout({xa: {"title": {"text": _xtitle(tv.trace.x_unit)}}})
    if window and hlink == "window":
        fig.update_xaxes(range=list(window))
    fig.update_layout(template=template(scheme), uirevision=uirevision, hovermode="x unified" if n == 1 else "closest",
                      showlegend=layout != "stacked", legend=dict(orientation="h", y=1.02, x=0),
                      margin=dict(l=70, r=20, t=30, b=50))
    if layout != "stacked":
        units_y = {v.trace.y_unit for v in views}
        fig.update_yaxes(title_text=", ".join(u for u in units_y if u))
        if views[0].scale in ("log", "loglog"):
            fig.update_yaxes(type="log")
    return fig


def spectra_figure(views: list[TraceView], kind: str = "fourier", scheme: str = "light", smooth: int = 0
                   ) -> go.Figure:
    from ..science.spectra import amplitude_spectrum_simple, psd_welch

    fig = go.Figure()
    for k, tv in enumerate(views):
        t = tv.trace
        if t.x_unit == "Hz" or t.n < 8:
            continue
        col = tv.color or PALETTE[k % len(PALETTE)]
        if kind == "psd":
            f, a = psd_welch(t.y, t.dt)
            ylab = f"PSD ({t.y_unit}²/Hz)" if t.y_unit else "PSD"
        else:
            f, a = amplitude_spectrum_simple(t.y, t.dt)
            ylab = f"Fourier amplitude ({t.y_unit}·s)" if t.y_unit else "Fourier amplitude"
            if smooth > 1:
                a = np.convolve(a, np.ones(smooth) / smooth, mode="same")
        f, a = f[1:], a[1:]
        fig.add_trace(go.Scatter(x=f, y=a, mode="lines", name=tv.label, line=dict(color=col, width=1.3),
                                 hovertemplate="%{x:.4g} Hz<br>%{y:.4g}<extra>" + tv.label + "</extra>"))
        fig.update_yaxes(title_text=ylab)
    fig.update_xaxes(type="log", title_text="frequency (Hz)")
    fig.update_yaxes(type="log", exponentformat="e")
    fig.update_layout(template=template(scheme), legend=dict(orientation="h", y=1.05), margin=dict(l=70, r=20, t=40,
                                                                                                  b=50))
    return fig


def spectrogram_figure(tv: TraceView | None, scheme: str = "light", cmap: str = "viridis") -> go.Figure:
    from ..science.colormaps import plotly_colorscale
    from ..science.spectra import spectrogram

    if tv is None or tv.trace.x_unit != "s":
        from .theme import empty_figure

        return empty_figure("Select a time-domain trace", scheme)
    f, t, S = spectrogram(tv.trace.y, tv.trace.dt)
    t = t + tv.trace.t0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.25, 0.75], vertical_spacing=0.03)
    fig.add_trace(go.Scattergl(x=tv.trace.x, y=tv.trace.y, mode="lines", line=dict(width=1, color=PALETTE[0]),
                               name=tv.label), row=1, col=1)
    vmax = float(np.nanmax(S))
    fig.add_trace(go.Heatmap(x=t, y=f, z=S.astype(np.float32), colorscale=plotly_colorscale(cmap), zmin=vmax - 80,
                             zmax=vmax, colorbar=dict(title="dB", len=0.7, y=0.37),
                             hovertemplate="t=%{x:.4g} s<br>f=%{y:.4g} Hz<br>%{z:.1f} dB<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(title_text="frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(template=template(scheme), showlegend=False, margin=dict(l=70, r=20, t=30, b=50),
                      title=dict(text=tv.label, font=dict(size=12), x=0.01))
    return fig


def response_figure(results: list[tuple[str, dict]], quantity: str = "psa", scheme: str = "light",
                    units: str = "") -> go.Figure:
    fig = go.Figure()
    for k, (label, r) in enumerate(results):
        fig.add_trace(go.Scatter(x=r["period"], y=r[quantity], mode="lines+markers", marker=dict(size=4),
                                 name=label, line=dict(color=PALETTE[k % len(PALETTE)])))
    names = {"psa": "pseudo-spectral acceleration", "psv": "pseudo-spectral velocity", "sd": "spectral displacement"}
    fig.update_xaxes(type="log", title_text="period (s)")
    fig.update_yaxes(type="log", title_text=f"{names[quantity]} {units}".strip(), exponentformat="e")
    fig.update_layout(template=template(scheme), legend=dict(orientation="h", y=1.05),
                      margin=dict(l=70, r=20, t=40, b=50))
    return fig


def record_section_figure(rows: list[tuple[float, Trace, str]], *, scale: float = 1.0, norm: str = "trace",
                          v_reduce: float = 0.0, arrivals: dict[str, float] | None = None, fill: bool = True,
                          scheme: str = "light") -> go.Figure:
    """Traces placed at their source distance (m); optional reduction velocity and arrival lines."""
    fig = go.Figure()
    if not rows:
        from .theme import empty_figure

        return empty_figure("No receivers with this variable", scheme)
    rows = sorted(rows, key=lambda r: r[0])
    dists = np.array([r[0] for r in rows])
    spacing = np.median(np.diff(np.unique(dists))) if len(np.unique(dists)) > 1 else (dists[0] or 1.0)
    gmax = max(np.nanmax(np.abs(t.y)) for _, t, _ in rows) or 1.0
    for d, t, label in rows:
        amp = (np.nanmax(np.abs(t.y)) or 1.0) if norm == "trace" else gmax
        y = d + scale * 0.6 * spacing * t.y / amp
        x = t.x - (d / v_reduce if v_reduce > 0 else 0.0)
        line_col = "#343a40" if scheme == "light" else "#dee2e6"
        if fill:   # variable-area: shade positive lobes between the baseline and the trace
            fig.add_trace(go.Scatter(x=x, y=np.full_like(x, d), mode="lines", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=np.maximum(y, d), mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(28,126,214,0.45)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=0.9, color=line_col), name=label,
                                 hovertemplate=f"{label}<br>distance {d:.4g} m<br>t=%{{x:.4g}} s<extra></extra>",
                                 showlegend=False))
    tmax = max(t.t_end for _, t, _ in rows)
    for name, v in (arrivals or {}).items():
        if v > 0:
            dd = np.linspace(0, dists.max() * 1.05, 50)
            tt = dd / v - (dd / v_reduce if v_reduce > 0 else 0.0)
            ok = tt <= tmax
            fig.add_trace(go.Scatter(x=tt[ok], y=dd[ok], mode="lines", name=f"{name} ({v:.0f} m/s)",
                                     line=dict(dash="dash", width=1.5, color="#e8590c" if name == "P" else "#1c7ed6")))
    xlab = "time (s)" if v_reduce <= 0 else f"reduced time t − d/{v_reduce:g} (s)"
    fig.update_layout(template=template(scheme), xaxis_title=xlab, yaxis_title="distance from source (m)",
                      legend=dict(orientation="h", y=1.05), margin=dict(l=70, r=20, t=40, b=50),
                      title=dict(text=f"amplitude: {'each trace normalised' if norm == 'trace' else 'common scale'}"
                                      f" (max {si_format(gmax, rows[0][1].y_unit)})", font=dict(size=11), x=0.01))
    return fig
