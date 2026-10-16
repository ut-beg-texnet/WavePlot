"""Geometry overlays (materials, sources, stopes, receivers) for 2-D plane figures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from ..io.geo import Geometry
from ..io.records import Extent
from .theme import geometry_colors

MATERIAL_FILLS = ["#adb5bd", "#74c0fc", "#8ce99a", "#ffd43b", "#ffa8a8", "#b197fc", "#63e6be", "#ffc078",
                  "#e599f7", "#a5d8ff"]
STOPE_TYPE_COLORS = {0: "#ff6b6b", 1: "#dee2e6", 2: "#e599f7", 3: "#ffd43b"}
HALO_DARK = "rgba(0,0,0,0.55)"
HALO_LIGHT = "rgba(255,255,255,0.9)"


@dataclass(frozen=True)
class PlaneSpec:
    ax1: str                     # horizontal grid axis
    ax2: str                     # vertical grid axis
    fixed_axis: str | None
    fixed_index: int | None
    spacing: dict[str, float]

    def span(self, ext: Extent, axis: str) -> tuple[float, float]:
        lo, hi = ext.range(axis)
        d = self.spacing[axis]
        if lo == hi:
            return (lo - 1) * d, (lo - 1) * d          # zero-thickness feature at node position
        return (lo - 1) * d, hi * d

    def intersects(self, ext: Extent, tol: int = 0) -> bool:
        if self.fixed_axis is None or self.fixed_index is None:
            return True
        lo, hi = ext.range(self.fixed_axis)
        return lo - tol <= self.fixed_index <= hi + tol

    def center(self, ijk: tuple[int, int, int], axis: str) -> float:
        idx = {"i": ijk[0], "j": ijk[1], "k": ijk[2]}[axis]
        return (idx - 0.5) * self.spacing[axis]


def _rect_path(x0: float, x1: float, y0: float, y1: float) -> tuple[list, list]:
    return [x0, x1, x1, x0, x0, None], [y0, y0, y1, y1, y0, None]


def geometry_traces(geom: Geometry | None, plane: PlaneSpec, layers: set[str], scheme: str = "light",
                    fill_materials: bool = False) -> list[go.BaseTraceType]:
    if geom is None:
        return []
    colors = geometry_colors(scheme)
    traces: list[go.BaseTraceType] = []
    a1, a2 = plane.ax1, plane.ax2

    if "materials" in layers and geom.materials:
        xs: list = []
        ys: list = []
        grid = geom.grid
        for m in geom.materials:
            if not plane.intersects(m.extent):
                continue
            x0, x1 = plane.span(m.extent, a1)
            y0, y1 = plane.span(m.extent, a2)
            whole = grid is not None and all(m.extent.range(a) == (grid.range(a)[0], grid.range(a)[1])
                                             for a in (a1, a2))
            if fill_materials and not whole:
                px, py = _rect_path(x0, x1, y0, y1)
                p = geom.prop_for(m.number)
                traces.append(go.Scatter(
                    x=px[:-1], y=py[:-1], fill="toself", mode="lines", line=dict(width=0),
                    fillcolor=MATERIAL_FILLS[(m.number - 1) % len(MATERIAL_FILLS)], opacity=0.35,
                    hoverinfo="text", showlegend=False,
                    text=f"material {m.number}" + (f"<br>vp {p.vp:.0f} m/s, vs {p.vs:.0f} m/s" if p else ""),
                    name=f"material {m.number}"))
            if not whole:
                px, py = _rect_path(x0, x1, y0, y1)
                xs += px
                ys += py
        if xs:
            # dark halo + light dashed stroke: visible on every colormap
            traces.append(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=HALO_DARK, width=3)))
            traces.append(go.Scatter(x=xs, y=ys, mode="lines", name="materials", hoverinfo="skip",
                                     line=dict(color=HALO_LIGHT, width=1.2, dash="dash"), showlegend=False))

    if "sources" in layers and geom.sources:
        sx, sy, txt = [], [], []
        for s in geom.sources:
            if not plane.intersects(s.extent, tol=1):
                continue
            x0, x1 = plane.span(s.extent, a1)
            y0, y1 = plane.span(s.extent, a2)
            sx.append((x0 + x1) / 2 if x0 != x1 else plane.center(_lo(s.extent), a1))
            sy.append((y0 + y1) / 2 if y0 != y1 else plane.center(_lo(s.extent), a2))
            txt.append(f"source {s.component}<br>i {s.extent.i1}-{s.extent.i2} j {s.extent.j1}-{s.extent.j2} "
                       f"k {s.extent.k1}-{s.extent.k2}")
        if sx:
            traces.append(go.Scatter(x=sx, y=sy, mode="markers", name="sources", text=txt, hoverinfo="text",
                                     marker=dict(symbol="star", size=13, color=colors["source"],
                                                 line=dict(color="white", width=1)), showlegend=False))

    if "stopes" in layers and geom.stopes:
        for st in geom.stopes:
            if not plane.intersects(st.extent):
                continue
            x0, x1 = plane.span(st.extent, a1)
            y0, y1 = plane.span(st.extent, a2)
            col = STOPE_TYPE_COLORS.get(st.stype, colors["stope"])
            info = (f"stope {st.stope_id} (type {st.stype})<br>i {st.extent.i1}-{st.extent.i2} "
                    f"j {st.extent.j1}-{st.extent.j2} k {st.extent.k1}-{st.extent.k2}")
            if "friction_deg" in st.params:
                info += f"<br>friction {st.params['friction_deg']:.3g}°, cohesion {st.params.get('cohesion', 0):.3g} Pa"
            px, py = _rect_path(x0, x1, y0, y1)
            traces.append(go.Scatter(x=px[:-1], y=py[:-1], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=HALO_DARK, width=6)))
            traces.append(go.Scatter(x=px[:-1], y=py[:-1], mode="lines", name=f"stope {st.stope_id}",
                                     line=dict(color=col, width=3 if st.stype == 2 or x0 == x1 or y0 == y1 else 2),
                                     hoverinfo="text", text=info, showlegend=False))
    return traces


def _lo(ext: Extent) -> tuple[int, int, int]:
    return ext.i1, ext.j1, ext.k1


def receiver_trace(receivers, plane: PlaneSpec, scheme: str = "light", max_offset_cells: float | None = None):
    """Receivers projected on the plane; filled if within one cell of the plane, hollow otherwise."""
    if not receivers:
        return None
    xs, ys, sym, txt, cdata = [], [], [], [], []
    for r in receivers:
        off = 0
        if plane.fixed_axis is not None and plane.fixed_index is not None:
            pos = {"i": r.position[0], "j": r.position[1], "k": r.position[2]}[plane.fixed_axis]
            off = pos - plane.fixed_index
            if max_offset_cells is not None and abs(off) > max_offset_cells:
                continue
        xs.append(plane.center(r.position, plane.ax1))
        ys.append(plane.center(r.position, plane.ax2))
        sym.append("triangle-down" if abs(off) <= 1 else "triangle-down-dot")
        d = plane.spacing.get(plane.fixed_axis or "i", 1.0) * abs(off)
        where = "on plane" if abs(off) <= 1 else f"{d:.4g} m off plane"
        txt.append(f"receiver R{r.number} ({r.position[0]},{r.position[1]},{r.position[2]}) — {where}"
                   f"<br>{', '.join(r.names)}<br><i>click to open histories</i>")
        cdata.append([r.number, *[h.number for h in r.records]][:8])
    if not xs:
        return None
    width = max(len(c) for c in cdata)
    cdata = [c + [None] * (width - len(c)) for c in cdata]
    return go.Scatter(x=xs, y=ys, mode="markers", name="receivers", text=txt, hoverinfo="text",
                      customdata=np.array(cdata, dtype=object),
                      marker=dict(symbol=sym, size=11, color="rgba(255,255,255,0.85)",
                                  line=dict(width=1.3, color="#111111")),
                      showlegend=False, meta="receivers")
