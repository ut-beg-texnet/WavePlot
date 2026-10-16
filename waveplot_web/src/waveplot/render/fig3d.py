"""3-D model view: grid box, material blocks, sources, stopes, receivers and snapshot planes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..io.records import Extent
from .overlays import MATERIAL_FILLS, STOPE_TYPE_COLORS
from .theme import template

AX_TO_SCENE = {"i": "x", "k": "y", "j": "z"}     # j (usually vertical) drawn upwards


def _bounds(ext: Extent, sp: dict[str, float]) -> dict[str, tuple[float, float]]:
    out = {}
    for a in "ijk":
        lo, hi = ext.range(a)
        out[a] = ((lo - 1) * sp[a], (lo - 1) * sp[a]) if lo == hi else ((lo - 1) * sp[a], hi * sp[a])
    return out


def _box_mesh(b: dict[str, tuple[float, float]], color: str, opacity: float, name: str, text: str) -> go.Mesh3d:
    xi, yk, zj = b["i"], b["k"], b["j"]
    x = [xi[0], xi[1], xi[1], xi[0], xi[0], xi[1], xi[1], xi[0]]
    y = [yk[0], yk[0], yk[1], yk[1], yk[0], yk[0], yk[1], yk[1]]
    z = [zj[0], zj[0], zj[0], zj[0], zj[1], zj[1], zj[1], zj[1]]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=name, hovertext=text,
                     hoverinfo="text", flatshading=True, showlegend=False)


def _box_edges(b: dict[str, tuple[float, float]]) -> tuple[list, list, list]:
    xi, yk, zj = b["i"], b["k"], b["j"]
    pts = [(x, y, z) for x in xi for y in yk for z in zj]
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    xs, ys, zs = [], [], []
    for a, c in edges:
        xs += [pts[a][0], pts[c][0], None]
        ys += [pts[a][1], pts[c][1], None]
        zs += [pts[a][2], pts[c][2], None]
    return xs, ys, zs


def model_figure(ds, layers: set[str], scheme: str = "light", plane_series=None, plane_frame: int = 0,
                 cmap: str = "balance", show_planes: bool = True) -> go.Figure:
    geom = ds.geometry
    fig = go.Figure()
    sp = ds.spacing
    grid = geom.grid if geom is not None and geom.grid is not None else None
    if grid is not None:
        b = _bounds(Extent(grid.i1, grid.i2 - 1, grid.j1, grid.j2 - 1, grid.k1, grid.k2 - 1), sp)
        xs, ys, zs = _box_edges(b)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#868e96", width=3),
                                   name="model", hoverinfo="skip", showlegend=False))
    if geom is not None and "materials" in layers:
        for m in geom.materials:
            if grid is not None and all(m.extent.range(a)[0] <= grid.range(a)[0] and
                                        m.extent.range(a)[1] >= grid.range(a)[1] - 1 for a in "ijk"):
                continue
            p = geom.prop_for(m.number)
            txt = f"material {m.number}" + (f"<br>vp {p.vp:.0f} m/s, vs {p.vs:.0f} m/s, ρ {p.density:.0f}" if p else "")
            fig.add_trace(_box_mesh(_bounds(m.extent, sp), MATERIAL_FILLS[(m.number - 1) % len(MATERIAL_FILLS)],
                                    0.18, f"material {m.number}", txt))
    if geom is not None and "stopes" in layers:
        for st in geom.stopes:
            txt = (f"stope {st.stope_id} (type {st.stype})<br>i {st.extent.i1}-{st.extent.i2} "
                   f"j {st.extent.j1}-{st.extent.j2} k {st.extent.k1}-{st.extent.k2}")
            fig.add_trace(_box_mesh(_bounds(st.extent, sp), STOPE_TYPE_COLORS.get(st.stype, "#e03131"), 0.75,
                                    f"stope {st.stope_id}", txt))
    if geom is not None and "sources" in layers and geom.sources:
        pts = [((s.extent.i1 - 0.5) * sp["i"], (s.extent.k1 - 0.5) * sp["k"], (s.extent.j1 - 0.5) * sp["j"],
                s.component) for s in geom.sources]
        fig.add_trace(go.Scatter3d(x=[p[0] for p in pts], y=[p[1] for p in pts], z=[p[2] for p in pts],
                                   mode="markers", marker=dict(size=7, color="#0b7285", symbol="diamond"),
                                   text=[f"source {p[3]}" for p in pts], hoverinfo="text", name="sources"))
    if "receivers" in layers and ds.receivers:
        r = ds.receivers
        fig.add_trace(go.Scatter3d(
            x=[(x.position[0] - 0.5) * sp["i"] for x in r], y=[(x.position[2] - 0.5) * sp["k"] for x in r],
            z=[(x.position[1] - 0.5) * sp["j"] for x in r], mode="markers",
            marker=dict(size=4, color="#212529" if scheme == "light" else "#f8f9fa"),
            text=[x.label for x in r], hoverinfo="text", name="receivers",
            customdata=[[x.number, *[h.number for h in x.records]][:8] + [None] * (8 - min(8, len(x.records) + 1))
                        for x in r]))
    if show_planes:
        _add_planes(fig, ds, plane_series, plane_frame, cmap, sp)
    fig.update_layout(
        template=template(scheme), margin=dict(l=0, r=0, t=10, b=0), uirevision="model3d",
        scene=dict(xaxis_title="x (m) [i]", yaxis_title="z (m) [k]", zaxis_title="y (m) [j]", aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))),
        legend=dict(orientation="h", y=1.02))
    return fig


def _add_planes(fig, ds, plane_series, frame, cmap, sp) -> None:
    from ..science.colormaps import plotly_colorscale
    from ..science.levels import robust_range

    if plane_series is None or plane_series.is_spacetime:
        return
    s = plane_series
    r = s.first
    a1, a2 = s.coords()
    z = s.frame(min(frame, len(s) - 1)).astype(np.float32)
    step = max(1, int(np.ceil(max(z.shape) / 160)))
    z = z[::step, ::step]
    c1, c2 = a1.coord[::step], a2.coord[::step]
    fixed = (r.fixed_index - 0.5) * sp[r.fixed_axis] if r.fixed_axis else 0.0
    A1, A2 = np.meshgrid(c1, c2, indexing="ij")
    coords = {r.ax1: A1, r.ax2: A2, r.fixed_axis: np.full_like(A1, fixed)}
    signed = bool(np.nanmin(z) < 0 < np.nanmax(z))
    lo, hi = robust_range(z, 99.5, symmetric=signed)
    fig.add_trace(go.Surface(
        x=coords["i"], y=coords["k"], z=coords["j"], surfacecolor=z,
        colorscale=plotly_colorscale(cmap if signed else "viridis"), cmin=lo, cmax=hi, opacity=0.9, showscale=True,
        colorbar=dict(title=s.units, len=0.6, thickness=12), name=s.name,
        hovertemplate=f"{s.name}: %{{surfacecolor:.4g}} {s.units}<extra></extra>"))
