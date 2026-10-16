"""Plotly figures for snapshot planes: colour map, contours and multi-line ("wiggle") views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

from ..science.colormaps import banded_colorscale, get_colors, is_diverging, plotly_colorscale
from ..science.compat import Provenance
from ..science.contours import contour_lines, pick_contour_levels
from ..science.decimate import coords_for, fit_grid
from ..science.levels import Levels, compute_levels, level_colors, robust_range
from .overlays import PlaneSpec, geometry_traces, receiver_trace
from .theme import geometry_colors, si_format, template


class SnapshotStyle(BaseModel):
    representation: Literal["map", "contour", "lines", "map+contour"] = "map"
    cmap: str = "balance"
    reverse: bool = False
    range_mode: Literal["frame", "series", "robust", "fixed"] = "robust"
    vmin: float | None = None
    vmax: float | None = None
    symmetric: bool = True
    banded: bool = False
    level_method: str = "lin"
    n_levels: int = Field(11, ge=2, le=256)
    abs_values: bool = False
    smooth: bool = False
    line_dir: Literal["h", "v", "both"] = "h"
    line_scale: float = 1.0
    line_color: bool = False
    max_lines: int = 80
    contour_engine: Literal["contourpy", "v407"] = "contourpy"
    overlays: list[str] = Field(default_factory=lambda: ["materials", "sources", "stopes", "receivers"])
    fill_materials: bool = False
    decimation: Literal["mean", "peak", "nearest"] = "mean"
    max_cells: int = 1400
    mode: Literal["standard", "v407"] = "standard"
    scheme: Literal["light", "dark"] = "light"
    show_title: bool = True


@dataclass
class PanelData:
    """One plane to draw: values ``z`` shaped ``(xqty, yqty)`` (rows along ``ax1``)."""

    z: np.ndarray
    x: np.ndarray                     # coordinates along ax1 (m or s)
    y: np.ndarray                     # coordinates along ax2
    ax1: str
    ax2: str
    title: str
    units: str = ""
    label: str = ""
    spacetime: bool = False
    vertical_up: bool = True
    plane: PlaneSpec | None = None
    series_range: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)


AXIS_NAMES = {"i": "x", "j": "y", "k": "z", "t": "t"}


def axis_title(axis: str) -> str:
    return "time (s)" if axis == "t" else f"{AXIS_NAMES.get(axis, axis)} (m)  [{axis}]"


def value_range(panel: PanelData, style: SnapshotStyle) -> tuple[float, float]:
    z = panel.z
    if style.range_mode == "fixed" and style.vmin is not None and style.vmax is not None:
        lo, hi = float(style.vmin), float(style.vmax)
    elif style.range_mode == "series" and panel.series_range is not None:
        lo, hi = panel.series_range
    elif style.range_mode == "robust":
        lo, hi = robust_range(z, 99.5)
    else:
        finite = z[np.isfinite(z)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if style.abs_values:
        lo, hi = 0.0, max(abs(lo), abs(hi))
    elif style.symmetric and lo < 0 < hi and is_diverging(style.cmap) and style.range_mode != "fixed":
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if hi == lo:
        hi = lo + (abs(lo) * 1e-3 or 1.0)
    return lo, hi


def make_levels(lo: float, hi: float, style: SnapshotStyle) -> Levels:
    return compute_levels(lo, hi, style.n_levels, style.level_method, abs_values=style.abs_values,
                          locked=style.range_mode == "fixed", mode=style.mode)


def _hover(panel: PanelData) -> str:
    u = f" {panel.units}" if panel.units else ""
    xn, yn = AXIS_NAMES.get(panel.ax1, panel.ax1), AXIS_NAMES.get(panel.ax2, panel.ax2)
    xu = "s" if panel.ax1 == "t" else "m"
    yu = "s" if panel.ax2 == "t" else "m"
    return f"{xn} = %{{x:.4g}} {xu}<br>{yn} = %{{y:.4g}} {yu}<br><b>%{{z:.4g}}{u}</b><extra></extra>"


def add_panel(fig: go.Figure, panel: PanelData, style: SnapshotStyle, row: int = 1, col: int = 1,
              geometry=None, receivers=None, coloraxis: str = "coloraxis", show_colorbar: bool = True,
              ) -> dict:
    """Add one plane to ``fig`` and return metadata (range, levels, provenance)."""
    prov = Provenance(mode=style.mode)
    lo, hi = value_range(panel, style)
    z = panel.z
    x, y = panel.x, panel.y
    nx, ny = z.shape
    if max(nx, ny) > style.max_cells:
        tx = min(nx, style.max_cells)
        ty = min(ny, style.max_cells)
        meth = "v407" if style.mode == "v407" else style.decimation
        z = fit_grid(z, (tx, ty), meth)
        x = coords_for(x, tx, meth)
        y = coords_for(y, ty, meth)
    zt = np.ascontiguousarray(z.T.astype(np.float32))       # (ny, nx): rows = vertical axis
    if style.abs_values:
        zt = np.abs(zt)

    levels = None
    rep = style.representation
    need_levels = style.banded or rep != "map" or style.mode == "v407"
    if need_levels:
        levels = make_levels(lo, hi, style)
        prov.add(*levels.provenance.quirks)
    colors = list(get_colors(style.cmap, 256, style.reverse))
    gc = geometry_colors(style.scheme)

    if rep in ("map", "map+contour"):
        banded = levels is not None and (style.banded or style.mode == "v407")
        if banded:
            zero_col = gc["zero"] if style.mode == "v407" else None
            lcols = level_colors(levels, colors, zero_col)
            zmin, zmax = min(lo, float(levels.values[0])), max(hi, float(levels.values[-1]))
            cs = banded_colorscale(levels.values, lcols, zmin, zmax,
                                   "toward_zero" if style.mode == "v407" else "floor")
        else:
            zmin, zmax = lo, hi
            cs = plotly_colorscale(style.cmap, style.reverse)
        fig.update_layout({coloraxis: dict(colorscale=cs, cmin=zmin, cmax=zmax, showscale=show_colorbar,
                                           colorbar=dict(title=dict(text=panel.units or "", side="right"),
                                                         thickness=14, len=0.9, exponentformat="e"))})
        fig.add_trace(go.Heatmap(z=zt, x=np.asarray(x, np.float32), y=np.asarray(y, np.float32),
                                 coloraxis=coloraxis, zsmooth="best" if style.smooth else False,
                                 hovertemplate=_hover(panel), name=panel.label or "value"), row=row, col=col)

    if rep in ("contour", "map+contour") and levels is not None:
        clev = pick_contour_levels(levels.values, 24 if style.mode == "v407" else 40)
        engine = "v407" if style.mode == "v407" else style.contour_engine
        lines = contour_lines(zt, np.asarray(x, float), np.asarray(y, float), clev, engine)
        lcols = level_colors(levels, colors)
        lvals = list(levels.values)
        for lev, xs, ys in lines:
            if xs.size == 0:
                continue
            k = int(np.argmin([abs(v - lev) for v in lvals]))
            col_line = lcols[k] if rep == "contour" else (gc["outline"])
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=col_line, width=1.3),
                                       hoverinfo="text", text=f"level {si_format(lev, panel.units)}",
                                       showlegend=False, connectgaps=False), row=row, col=col)
        if rep == "contour":
            fig.update_layout({coloraxis: dict(colorscale=plotly_colorscale(style.cmap, style.reverse), cmin=lo,
                                               cmax=hi, showscale=show_colorbar,
                                               colorbar=dict(title=dict(text=panel.units or ""), thickness=14))})
            # invisible marker trace to carry the colorbar
            fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers",
                                     marker=dict(size=0.1, color=[lo], coloraxis=coloraxis, opacity=0),
                                     hoverinfo="skip", showlegend=False), row=row, col=col)
        if engine == "v407":
            prov.add("v407_contours")

    if rep == "lines":
        _add_lines(fig, panel, zt, np.asarray(x, float), np.asarray(y, float), style, levels, colors, lo, hi,
                   row, col)

    if panel.plane is not None and not panel.spacetime:
        for tr in geometry_traces(geometry, panel.plane, set(style.overlays), style.scheme, style.fill_materials):
            fig.add_trace(tr, row=row, col=col)
        if "receivers" in style.overlays and receivers:
            rt = receiver_trace(receivers, panel.plane, style.scheme)
            if rt is not None:
                fig.add_trace(rt, row=row, col=col)
    return {"range": (lo, hi), "levels": levels, "provenance": prov}


def _add_lines(fig, panel, zt, x, y, style, levels, colors, lo, hi, row, col):
    """Multi-line representation: each row/column displaced in proportion to its values."""
    amax = max(abs(lo), abs(hi)) or 1.0
    ny, nx = zt.shape
    dirs = ["h", "v"] if style.line_dir == "both" else [style.line_dir]
    gc = geometry_colors(style.scheme)
    for d in dirs:
        if d == "h":
            idx = np.unique(np.linspace(0, ny - 1, min(ny, style.max_lines)).round().astype(int))
            spacing = abs(y[idx[1]] - y[idx[0]]) if len(idx) > 1 else 1.0
            scale = style.line_scale * spacing / amax
            sign = 1.0 if panel.vertical_up else -1.0
            xs = np.concatenate([np.append(x, np.nan) for _ in idx])
            ys = np.concatenate([np.append(y[r] + sign * scale * zt[r], np.nan) for r in idx])
        else:
            idx = np.unique(np.linspace(0, nx - 1, min(nx, style.max_lines)).round().astype(int))
            spacing = abs(x[idx[1]] - x[idx[0]]) if len(idx) > 1 else 1.0
            scale = style.line_scale * spacing / amax
            xs = np.concatenate([np.append(x[c] + scale * zt[:, c], np.nan) for c in idx])
            ys = np.concatenate([np.append(y, np.nan) for _ in idx])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(width=1, color=gc["outline"]),
                                   hoverinfo="skip", showlegend=False), row=row, col=col)
    fig.add_annotation(text=f"line scale: max |value| {si_format(amax, panel.units)} = "
                            f"{style.line_scale:g} line spacing",
                       xref="paper", yref="paper", x=0, y=-0.12, showarrow=False, font=dict(size=11),
                       xanchor="left")


def _edges(c: np.ndarray) -> list[float]:
    if len(c) < 2:
        return [float(c[0]) - 0.5, float(c[0]) + 0.5] if len(c) else [0.0, 1.0]
    return [float(c[0] - (c[1] - c[0]) / 2), float(c[-1] + (c[-1] - c[-2]) / 2)]


def axis_layout(panel: PanelData, true_scale: bool, anchor: str | None = None) -> tuple[dict, dict]:
    xr, yr = _edges(np.asarray(panel.x, float)), _edges(np.asarray(panel.y, float))
    xa = dict(title=dict(text=axis_title(panel.ax1)), showgrid=False, zeroline=False, constrain="domain", range=xr)
    ya = dict(title=dict(text=axis_title(panel.ax2)), showgrid=False, zeroline=False, constrain="domain",
              range=yr if panel.vertical_up else yr[::-1])
    if true_scale and not panel.spacetime:
        ya["scaleanchor"] = anchor or "x"
        ya["scaleratio"] = 1
    return xa, ya


def snapshot_figure(panels: list[PanelData], style: SnapshotStyle, *, geometry=None, receivers=None,
                    rows: int = 1, cols: int = 1, linked: bool = True, uirevision: str = "snap",
                    title: str | None = None) -> tuple[go.Figure, list[dict]]:
    """Figure with one or more planes (multi-panel grids share the colour axis when ``linked``)."""
    if rows * cols > 1:
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p.title for p in panels],
                            horizontal_spacing=0.06, vertical_spacing=0.1)
    else:
        fig = make_subplots(rows=1, cols=1)
    metas = []
    if linked and len(panels) > 1:
        lo = min(value_range(p, style)[0] for p in panels)
        hi = max(value_range(p, style)[1] for p in panels)
        style = style.model_copy(update={"range_mode": "fixed", "vmin": lo, "vmax": hi})
    for n, p in enumerate(panels):
        r, c = n // cols + 1, n % cols + 1
        ca = "coloraxis" if (linked or n == 0) else f"coloraxis{n + 1}"
        meta = add_panel(fig, p, style, r, c, geometry, receivers, coloraxis=ca,
                         show_colorbar=(n == 0 or not linked))
        metas.append(meta)
        xa, ya = axis_layout(p, True, anchor="x" if n == 0 else f"x{n + 1}")
        sx = "xaxis" if n == 0 else f"xaxis{n + 1}"
        sy = "yaxis" if n == 0 else f"yaxis{n + 1}"
        if rows * cols > 1:
            xa["title"]["text"] = xa["title"]["text"] if r == rows else ""
            ya["title"]["text"] = ya["title"]["text"] if c == 1 else ""
        fig.update_layout({sx: xa, sy: ya})
        if not linked and n > 0:
            dom = fig.layout[sx].domain
            fig.update_layout({ca: dict(colorbar=dict(x=dom[1] + 0.005, len=0.8 / rows, thickness=10))})
    fig.update_layout(template=template(style.scheme), uirevision=uirevision, dragmode="zoom",
                      margin=dict(l=70, r=30, t=50 if (title or rows * cols > 1) else 30, b=60),
                      title=dict(text=title, x=0.01, font=dict(size=14)) if (title and style.show_title) else None,
                      clickmode="event", hovermode="closest")
    return fig, metas
