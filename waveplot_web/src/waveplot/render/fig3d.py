"""3-D model view: grid box, material blocks, sources, stopes, receivers and snapshot planes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..io.records import Extent
from .overlays import MATERIAL_FILLS, STOPE_TYPE_COLORS, stope_hover
from .theme import template

AX_TO_SCENE = {"i": "x", "k": "y", "j": "z"}     # j (usually vertical) drawn upwards

_UP_Z = dict(x=0, y=0, z=1)
_CENTER = dict(x=0, y=0, z=0)
_ORTHO = dict(type="orthographic")
# Scene axes: x = i, y = k, z = j (up). Front / side / top match the i-j, k-j and i-k cross-sections.
CAMERAS = {
    "iso": dict(eye=dict(x=1.45, y=-2.05, z=1.0), up=_UP_Z, center=_CENTER, projection=dict(type="perspective")),
    "front": dict(eye=dict(x=0, y=-2.2, z=0), up=_UP_Z, center=_CENTER, projection=_ORTHO),
    "side": dict(eye=dict(x=2.2, y=0, z=0), up=_UP_Z, center=_CENTER, projection=_ORTHO),
    "top": dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0), center=_CENTER, projection=_ORTHO),
}
# scene axis seen end-on in each flat view: its labels would pile up in one corner
_END_ON = {"front": "yaxis", "side": "xaxis", "top": "zaxis"}


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
    """Line coordinates of a box's edges; a flat box (a plane) gives just its 4-edge outline."""
    xi, yk, zj = b["i"], b["k"], b["j"]
    pts = [(x, y, z) for x in xi for y in yk for z in zj]
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    xs, ys, zs, seen = [], [], [], set()
    for a, c in edges:
        key = frozenset((pts[a], pts[c]))
        if pts[a] == pts[c] or key in seen:
            continue
        seen.add(key)
        xs += [pts[a][0], pts[c][0], None]
        ys += [pts[a][1], pts[c][1], None]
        zs += [pts[a][2], pts[c][2], None]
    return xs, ys, zs


def _darken(hex_color: str, f: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[n:n + 2], 16) for n in (0, 2, 4))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


def _stope_plane(st, sp: dict[str, float]) -> tuple[str, float] | None:
    """Fixed axis and coordinate (m) of a planar stope: the grid face ``(lo-1)·d``, as drawn by ``_bounds``."""
    for a in "ijk":
        lo, hi = st.extent.range(a)
        if lo == hi:
            return a, (lo - 1) * sp[a]
    return None


def model_figure(ds, layers: set[str], scheme: str = "light", plane_series=None, plane_frame: int = 0,
                 cmap: str = "balance", show_planes: bool = True, camera: str = "iso") -> go.Figure:
    geom = ds.geometry
    fig = go.Figure()
    sp = ds.spacing
    grid = geom.grid if geom is not None and geom.grid is not None else None
    if grid is not None:
        ext = Extent(grid.i1, grid.i2 - 1, grid.j1, grid.j2 - 1, grid.k1, grid.k2 - 1)
        b = _bounds(ext, sp)
        xs, ys, zs = _box_edges(b)
        txt = "model<br>" + "<br>".join(
            f"{a} {ext.range(a)[0]}–{ext.range(a)[1]} ({b[a][0]:g}–{b[a][1]:g} m)" for a in "ijk")
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#868e96", width=3),
                                   name="model", text=txt, hoverinfo="text", showlegend=False))
    plane = None
    if show_planes and plane_series is not None and not plane_series.is_spacetime:
        plane = _plane_trace(ds, plane_series, plane_frame, cmap, sp)
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
            b = _bounds(st.extent, sp)
            col = STOPE_TYPE_COLORS.get(st.stype, "#e03131")
            txt = stope_hover(st)
            on = _stope_plane(st, sp)
            # a snapshot drawn on (or within a cell of) this stope's plane replaces its fill: outline only
            draped = (plane is not None and on is not None and plane[1] == on[0]
                      and abs(plane[2] - on[1]) <= sp[on[0]] + 1e-9)
            if not draped:
                fig.add_trace(_box_mesh(b, col, 0.75, f"stope {st.stope_id}", txt))
            xs, ys, zs = _box_edges(b)
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=_darken(col), width=5),
                                       name=f"stope {st.stope_id} outline", text=txt, hoverinfo="text",
                                       showlegend=False))
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
    if plane is not None:
        fig.add_trace(plane[0])
    camera = camera if camera in CAMERAS else "iso"
    fig.update_layout(
        template=template(scheme), margin=dict(l=0, r=0, t=10, b=0), uirevision=f"model3d-{camera}",
        scene=dict(xaxis_title="x (m) [i]", yaxis_title="z (m) [k]", zaxis_title="y (m) [j]", aspectmode="data",
                   camera=CAMERAS[camera]),
        legend=dict(orientation="h", y=1.02))
    if camera in _END_ON:
        fig.update_layout(scene={_END_ON[camera]: dict(title_text="", showticklabels=False)})
    return fig


def _plane_trace(ds, s, frame, cmap, sp) -> tuple[go.Surface, str, float] | None:
    """A snapshot frame drawn as a surface on its plane: (trace, fixed axis, fixed coordinate in m)."""
    from ..science.colormaps import plotly_colorscale
    from ..science.levels import robust_range

    r = s.first
    if not r.fixed_axis:
        return None
    a1, a2 = s.coords()
    z = s.frame(min(frame, len(s) - 1)).astype(np.float32)
    step = max(1, int(np.ceil(max(z.shape) / 160)))
    z = z[::step, ::step]
    fixed = (r.fixed_index - 0.5) * sp[r.fixed_axis]
    geom = ds.geometry
    if s.is_stope and s.stope_id is not None and geom is not None:
        # fault quantities live on the fault itself: draw them exactly where the stope is drawn
        st = next((x for x in geom.stopes if x.stope_id == s.stope_id), None)
        on = _stope_plane(st, sp) if st is not None else None
        if on is not None and on[0] == r.fixed_axis:
            fixed = on[1]
    A1, A2 = np.meshgrid(a1.coord[::step], a2.coord[::step], indexing="ij")
    coords = {r.ax1: A1, r.ax2: A2, r.fixed_axis: np.full_like(A1, fixed)}
    I1, I2 = np.meshgrid(a1.index[::step], a2.index[::step], indexing="ij")
    idx = {r.ax1: I1, r.ax2: I2, r.fixed_axis: np.full_like(I1, r.fixed_index)}
    # 2-D object array: plotly stringifies the rows of a nested list
    text = np.array([[f"i {i} · j {j} · k {k}" for i, j, k in zip(ri, rj, rk, strict=True)]
                     for ri, rj, rk in zip(idx["i"], idx["j"], idx["k"], strict=True)], dtype=object)
    signed = bool(np.nanmin(z) < 0 < np.nanmax(z))
    lo, hi = robust_range(z, 99.5, symmetric=signed)
    title = s.units
    if not hi > lo:     # e.g. TaSlip with slip=off: say so rather than show plotly's arbitrary default range
        title = f"{s.units}<br>constant<br>{lo:.4g}"
        pad = abs(lo) * 0.01 or 1.0
        lo, hi = lo - pad, hi + pad
    trace = go.Surface(
        x=coords["i"], y=coords["k"], z=coords["j"], surfacecolor=z, text=text,
        colorscale=plotly_colorscale(cmap if signed else "viridis"), cmin=lo, cmax=hi, opacity=0.9, showscale=True,
        colorbar=dict(title=title, len=0.6, thickness=12), name=s.name,
        hovertemplate=(f"%{{text}}<br>x %{{x:.1f}} m · y %{{z:.1f}} m · z %{{y:.1f}} m"
                       f"<br>{s.name}: %{{surfacecolor:.4g}} {s.units}<extra></extra>"))
    return trace, r.fixed_axis, fixed
