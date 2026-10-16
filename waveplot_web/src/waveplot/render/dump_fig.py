"""Glyph traces for dumps: principal-stress crosses and velocity arrows; Mohr-circle figures."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from ..science import stress
from ..science.decimate import dump_glyph_indices
from .theme import si_format, template


def _robust_max(mags: np.ndarray, mode: str) -> float:
    """Reference magnitude for glyph lengths: v407 used the maximum; standard the 98th percentile."""
    m = mags[np.isfinite(mags)]
    if m.size == 0:
        return 1.0
    v = float(m.max()) if mode == "v407" else float(np.percentile(m, 98))
    return v or float(m.max()) or 1.0


def _grid(x: np.ndarray, y: np.ndarray, nmax: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    return dump_glyph_indices(len(x), min(nmax, len(x)), mode), dump_glyph_indices(len(y), min(nmax, len(y)), mode)


def principal_glyphs(x, y, s11, s22, s12, nmax: int = 51, scale: float = 1.0, mode: str = "standard",
                     units: str = "Pa") -> tuple[list[go.BaseTraceType], str]:
    """Crossed bars along the principal directions; length ∝ |σ|; compression and tension coloured."""
    ix, iy = _grid(x, y, nmax, mode)
    S11, S22, S12 = (np.asarray(a)[np.ix_(ix, iy)] for a in (s11, s22, s12))
    p = stress.principal_2d(S11, S22, S12, mode)
    X, Y = np.meshgrid(np.asarray(x)[ix], np.asarray(y)[iy], indexing="ij")
    dx = abs(x[ix[1]] - x[ix[0]]) if len(ix) > 1 else 1.0
    dy = abs(y[iy[1]] - y[iy[0]]) if len(iy) > 1 else dx
    spacing = min(dx, dy)
    mags = np.maximum(np.abs(p.sigma_max), np.abs(p.sigma_min))
    vmax = _robust_max(mags, mode)
    k = scale * 0.5 * spacing / vmax            # half-length per unit stress (robust max bar = spacing)
    comp_x, comp_y, ten_x, ten_y = [], [], [], []
    for val, ang in ((p.sigma_min, p.angle_min), (p.sigma_max, p.angle_min + np.pi / 2)):
        L = np.minimum(k * np.abs(val), 0.75 * spacing * max(scale, 1.0))   # clip isolated extremes
        ex, ey = L * np.cos(ang), L * np.sin(ang)
        segx = np.stack([X - ex, X + ex, np.full_like(X, np.nan)], axis=-1)
        segy = np.stack([Y - ey, Y + ey, np.full_like(Y, np.nan)], axis=-1)
        compressive = val <= 0 if mode == "v407" else val < 0
        comp_x.append(segx[compressive].ravel())
        comp_y.append(segy[compressive].ravel())
        ten_x.append(segx[~compressive].ravel())
        ten_y.append(segy[~compressive].ravel())
    traces = [
        go.Scattergl(x=np.concatenate(comp_x), y=np.concatenate(comp_y), mode="lines", name="compression",
                     line=dict(color="#1971c2", width=1.6), hoverinfo="skip"),
        go.Scattergl(x=np.concatenate(ten_x), y=np.concatenate(ten_y), mode="lines", name="tension",
                     line=dict(color="#e03131", width=2.6), hoverinfo="skip"),
    ]
    note = f"bar length: one glyph spacing = {si_format(vmax / scale, units)} (longer bars clipped)"
    return traces, note


def velocity_arrows(x, y, u, v, nmax: int = 41, scale: float = 1.0, mode: str = "standard",
                    color: str = "#e03131") -> tuple[list[go.BaseTraceType], str]:
    ix, iy = _grid(x, y, nmax, mode)
    U, V = (np.asarray(a)[np.ix_(ix, iy)] for a in (u, v))
    X, Y = np.meshgrid(np.asarray(x)[ix], np.asarray(y)[iy], indexing="ij")
    dx = abs(x[ix[1]] - x[ix[0]]) if len(ix) > 1 else 1.0
    dy = abs(y[iy[1]] - y[iy[0]]) if len(iy) > 1 else dx
    spacing = min(dx, dy)
    mag = np.hypot(U, V)
    vmax = _robust_max(mag, mode)
    k = scale * spacing / vmax
    clip = np.minimum(1.0, 1.5 * spacing * max(scale, 1.0) / np.maximum(mag * k, 1e-300))
    ax, ay = U * k * clip, V * k * clip       # full arrow, centred on the node (v407 screen)
    x0, y0, x1, y1 = X - ax / 2, Y - ay / 2, X + ax / 2, Y + ay / 2
    hl = np.maximum(np.hypot(ax, ay) / 4, 0.0)
    ang = np.arctan2(ay, ax)
    hx1, hy1 = x1 - hl * np.cos(ang - 0.45), y1 - hl * np.sin(ang - 0.45)
    hx2, hy2 = x1 - hl * np.cos(ang + 0.45), y1 - hl * np.sin(ang + 0.45)
    nan = np.full_like(X, np.nan)
    xs = np.stack([x0, x1, nan, hx1, x1, hx2, nan], axis=-1).ravel()
    ys = np.stack([y0, y1, nan, hy1, y1, hy2, nan], axis=-1).ravel()
    tr = go.Scattergl(x=xs, y=ys, mode="lines", name="velocity", line=dict(color=color, width=1.4),
                      hoverinfo="skip")
    return [tr], f"arrow length: one glyph spacing = {si_format(vmax / scale, 'm/s')}"


def mohr_figure(m: stress.MohrCircles, friction: float | None = None, cohesion: float = 0.0, scheme: str = "light",
                title: str = "") -> go.Figure:
    fig = go.Figure()
    th = np.linspace(0, 2 * math.pi, 181)
    cols = ["#1c7ed6", "#868e96", "#868e96"]
    lo = min(c - r for c, r in m.circles)
    hi = max(c + r for c, r in m.circles)
    for n, (c, r) in enumerate(m.circles):
        fig.add_trace(go.Scatter(x=c + r * np.cos(th), y=r * np.sin(th), mode="lines",
                                 line=dict(color=cols[n % 3], width=2 if n == 0 else 1), hoverinfo="skip",
                                 showlegend=False))
    for sn, tau, lab in m.points:
        fig.add_trace(go.Scatter(x=[sn], y=[tau], mode="markers+text", text=[lab], textposition="top center",
                                 marker=dict(size=8, color="#e8590c"), showlegend=False,
                                 hovertemplate="σn %{x:.4g} Pa<br>τ %{y:.4g} Pa<extra></extra>"))
    if friction is not None:
        sn, tau = stress.failure_line(friction, cohesion, (lo, hi))
        fig.add_trace(go.Scatter(x=sn, y=tau, mode="lines", name=f"Coulomb φ={friction:g}°",
                                 line=dict(color="#c92a2a", dash="dash")))
        fig.add_trace(go.Scatter(x=sn, y=-tau, mode="lines", showlegend=False, line=dict(color="#c92a2a",
                                                                                          dash="dash")))
    fig.update_layout(template=template(scheme), xaxis_title="normal stress σn (Pa, tension +)",
                      yaxis_title="shear τ (Pa)", margin=dict(l=60, r=10, t=30, b=45),
                      title=dict(text=title, font=dict(size=11), x=0.01), legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
