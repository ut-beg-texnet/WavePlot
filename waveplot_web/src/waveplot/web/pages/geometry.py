"""Geometry page: interactive 3-D model, 2-D cross-sections, particle motion and crack dumps."""

from __future__ import annotations

import time

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from ...render.fig3d import model_figure
from ...render.overlays import PlaneSpec
from ...render.snapshot_fig import PanelData, SnapshotStyle, snapshot_figure
from ...render.theme import PALETTE, empty_figure, template
from ...science.trajectories import trajectory
from .. import state
from ..components import kv_table, section, simple_table

LAYERS = [("materials", "Materials"), ("sources", "Sources"), ("stopes", "Stopes / faults"),
          ("receivers", "Receivers")]
CAMERA_VIEWS = [("iso", "Iso"), ("top", "Top"), ("front", "Front (i-j)"), ("side", "Side (k-j)")]


def layout(**_kwargs):
    controls = dmc.Stack([
        section("3-D view"),
        dmc.CheckboxGroup(id="geo-layers", value=[k for k, _ in LAYERS], children=dmc.SimpleGrid(
            [dmc.Checkbox(label=lab, value=k) for k, lab in LAYERS], cols=2, spacing=4, verticalSpacing=4)),
        dmc.SegmentedControl(id="geo-camera", data=[{"label": lab, "value": v} for v, lab in CAMERA_VIEWS],
                             value="iso", fullWidth=True, size="xs"),
        dmc.Select(id="geo-plane", label="Show snapshot on its plane", data=[], clearable=True,
                   placeholder="none", comboboxProps={"withinPortal": True}),
        dmc.NumberInput(id="geo-plane-frame", label="Frame", value=0, min=0, w=120),
        section("Cross-section"),
        dmc.Group([
            dmc.SegmentedControl(id="geo-sec-axis", data=[{"label": "i-j (k fixed)", "value": "k"},
                                                          {"label": "i-k (j fixed)", "value": "j"},
                                                          {"label": "k-j (i fixed)", "value": "i"}], value="k"),
        ]),
        dmc.NumberInput(id="geo-sec-index", label="Fixed index", value=1, min=1, w=140),
        dmc.Switch(id="geo-sec-fill", label="Fill materials", checked=True),
        section("Particle motion"),
        dmc.Select(id="geo-traj-rec", label="Receiver", data=[], comboboxProps={"withinPortal": True},
                   placeholder="receivers with velocity histories"),
        dmc.SegmentedControl(id="geo-traj-kind", data=[{"label": "Displacement", "value": "disp"},
                                                       {"label": "Velocity", "value": "vel"},
                                                       {"label": "Acceleration", "value": "acc"}], value="disp",
                             fullWidth=True),
    ], gap=6)
    tabs = dmc.Tabs([
        dmc.TabsList([dmc.TabsTab("3-D model", value="3d"), dmc.TabsTab("Cross-section", value="2d"),
                      dmc.TabsTab("Particle motion", value="traj"), dmc.TabsTab("Crack dumps", value="crk"),
                      dmc.TabsTab("Model data", value="info")]),
        dmc.TabsPanel(dcc.Loading(dcc.Graph(id="geo-3d", className="wp-graph", config={"displaylogo": False}),
                                  type="dot", delay_show=400), value="3d"),
        dmc.TabsPanel(dcc.Graph(id="geo-2d", className="wp-graph", config={"displaylogo": False}), value="2d"),
        dmc.TabsPanel(dcc.Graph(id="geo-traj", className="wp-graph", config={"displaylogo": False}), value="traj"),
        dmc.TabsPanel([
            dmc.Group([dmc.Select(id="geo-crk", data=[], placeholder="crack dump", w=260,
                                  comboboxProps={"withinPortal": True}),
                       dmc.Badge("format unverified (no sample file)", color="yellow", variant="light", tt="none")],
                      mt=4),
            dcc.Graph(id="geo-crk-graph", className="wp-graph-short", config={"displaylogo": False}),
        ], value="crk"),
        dmc.TabsPanel(html.Div(id="geo-info", style={"marginTop": 8}), value="info"),
    ], value="3d")
    return dmc.Grid([
        dmc.GridCol(dmc.Paper(controls, className="wp-controls"), span={"base": 12, "md": 3}),
        dmc.GridCol(dmc.Paper(tabs), span={"base": 12, "md": 9}),
    ], gutter="sm")


def register_callbacks(app) -> None:
    @app.callback(Output("geo-plane", "data"), Output("geo-traj-rec", "data"), Output("geo-traj-rec", "value"),
                  Output("geo-crk", "data"), Output("geo-sec-index", "value"), Output("geo-sec-index", "max"),
                  Input("store-ds", "data"), Input("geo-sec-axis", "value"))
    def _options(ds_id, axis):
        ds = state.get_dataset(ds_id)
        if ds is None:
            return [], [], None, [], no_update, no_update
        planes = [{"value": s.slug, "label": (f"fault {s.stope_id}: " if s.stope_id is not None else "") + s.label}
                  for s in ds.series.values() if not s.is_spacetime]
        recs = [{"value": str(r.number), "label": r.label} for r in ds.receivers if r.velocity]
        crk = [{"value": str(c.number), "label": f"crack dump #{c.number} (t = {c.time:.4g} s)"} for c in ds.cracks]
        geom = ds.geometry
        if geom is not None and geom.grid is not None:
            lo, hi = geom.grid.range(axis or "k")
            mid = (lo + hi) // 2
            src = geom.sources[0].extent.range(axis or "k")[0] if geom.sources else mid
            return planes, recs, (recs[0]["value"] if recs else None), crk, src, hi
        return planes, recs, (recs[0]["value"] if recs else None), crk, no_update, no_update

    @app.callback(Output("geo-3d", "figure"), Input("store-ds", "data"), Input("geo-layers", "value"),
                  Input("geo-plane", "value"), Input("geo-plane-frame", "value"), Input("geo-camera", "value"),
                  Input("store-prefs", "data"))
    def _model(ds_id, layers, plane, frame, camera, prefs):
        scheme = state.scheme_of(prefs)
        ds = state.get_dataset(ds_id)
        if ds is None:
            return empty_figure("Select a dataset", scheme)
        if ds.geometry is None and not ds.receivers:
            return empty_figure("This dataset has no geometry", scheme)
        s = ds.series.get(plane) if plane else None
        return model_figure(ds, set(layers or []), scheme, s, int(frame or 0), camera=camera or "iso")

    @app.callback(Output("url", "pathname", allow_duplicate=True), Output("store-nav", "data", allow_duplicate=True),
                  Input("geo-3d", "clickData"), State("store-ds", "data"), prevent_initial_call=True)
    def _open_receiver(click, ds_id):
        if not click or not click.get("points"):
            return no_update, no_update
        cd = click["points"][0].get("customdata")
        if not cd or len(cd) < 2:
            return no_update, no_update
        return "/histories", {"open_histories": [int(h) for h in cd[1:] if h is not None], "ds": ds_id,
                              "ts": time.time()}

    @app.callback(Output("geo-2d", "figure"), Input("store-ds", "data"), Input("geo-sec-axis", "value"),
                  Input("geo-sec-index", "value"), Input("geo-sec-fill", "checked"), Input("geo-layers", "value"),
                  Input("store-prefs", "data"))
    def _section(ds_id, axis, index, fill, layers, prefs):
        scheme = state.scheme_of(prefs)
        ds = state.get_dataset(ds_id)
        geom = ds.geometry if ds else None
        if geom is None or geom.grid is None:
            return empty_figure("No geometry", scheme)
        axis = axis or "k"
        ax1, ax2 = {"k": ("i", "j"), "j": ("i", "k"), "i": ("k", "j")}[axis]
        sp = ds.spacing
        g = geom.grid
        n1 = g.range(ax1)[1] - g.range(ax1)[0]
        n2 = g.range(ax2)[1] - g.range(ax2)[0]
        x = (np.arange(n1) + 0.5) * sp[ax1]
        y = (np.arange(n2) + 0.5) * sp[ax2]
        # material number raster (last definition wins, as WAVE applies them in order)
        z = np.full((n1, n2), np.nan, np.float32)
        idx = int(index or 1)
        for m in geom.materials:
            if not (m.extent.range(axis)[0] <= idx <= m.extent.range(axis)[1]):
                continue
            a0, a1_ = m.extent.range(ax1)
            b0, b1 = m.extent.range(ax2)
            p = geom.prop_for(m.number)
            z[max(a0 - 1, 0):a1_, max(b0 - 1, 0):b1] = p.vp if p else m.number
        plane = PlaneSpec(ax1, ax2, axis, idx, sp)
        panel = PanelData(z=z, x=x, y=y, ax1=ax1, ax2=ax2, title="", units="m/s", label="vp",
                          vertical_up=ax2 == "j", plane=plane)
        style = SnapshotStyle(cmap="viridis", range_mode="frame", symmetric=False, overlays=list(layers or []),
                              scheme=scheme, representation="map" if fill else "contour", n_levels=2)
        fig, _ = snapshot_figure([panel], style, geometry=geom, receivers=ds.receivers,
                                 uirevision=f"geo2d{axis}")
        fig.update_layout(title=dict(text=f"P-wave speed on the {ax1}-{ax2} plane at {axis} = {idx}", x=0.01,
                                     font=dict(size=13)))
        return fig

    @app.callback(Output("geo-traj", "figure"), Input("store-ds", "data"), Input("geo-traj-rec", "value"),
                  Input("geo-traj-kind", "value"), Input("store-prefs", "data"))
    def _traj(ds_id, rec, kind, prefs):
        scheme, mode = state.scheme_of(prefs), state.mode_of(prefs)
        ds = state.get_dataset(ds_id)
        if ds is None:
            return empty_figure("Select a dataset", scheme)
        r = next((x for x in ds.receivers if str(x.number) == str(rec)), None)
        if r is None or r.velocity is None:
            return empty_figure("Particle motion needs receivers with XVEL/YVEL/ZVEL histories — none in this run",
                                scheme)
        comps = [ds.history_data(h.number) if h is not None else None for h in r.velocity]
        dt = next(h.dt for h in r.velocity if h is not None)
        path = trajectory(*comps, dt=dt, kind=kind or "disp", mode=mode)
        t = np.arange(len(path)) * dt
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=3, specs=[[{}, {}, {}], [{"type": "scene", "colspan": 3}, None, None]],
                            subplot_titles=["x–y", "x–z", "z–y", "3-D path (colour = time)"], row_heights=[0.4, 0.6])
        pairs = [(0, 1), (0, 2), (2, 1)]
        names = ["x", "y", "z"]
        for c, (a, b) in enumerate(pairs, start=1):
            fig.add_trace(go.Scattergl(x=path[:, a], y=path[:, b], mode="markers+lines",
                                       marker=dict(size=3, color=t, colorscale="Viridis"), line=dict(width=0.5,
                                                                                                    color="#adb5bd"),
                                       showlegend=False), row=1, col=c)
            fig.update_xaxes(title_text=names[a], row=1, col=c)
            fig.update_yaxes(title_text=names[b], row=1, col=c, scaleanchor=f"x{c}" if c > 1 else "x")
        fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 2], z=path[:, 1], mode="lines",
                                   line=dict(color=t, colorscale="Viridis", width=4), showlegend=False), row=2, col=1)
        fig.update_layout(template=template(scheme), margin=dict(l=40, r=10, t=40, b=10),
                          title=dict(text=f"{r.label} — {kind}", x=0.01, font=dict(size=12)))
        return fig

    @app.callback(Output("geo-crk-graph", "figure"), Input("geo-crk", "value"), Input("store-prefs", "data"),
                  State("store-ds", "data"))
    def _crk(num, prefs, ds_id):
        scheme = state.scheme_of(prefs)
        ds = state.get_dataset(ds_id)
        if ds is None or not ds.cracks:
            return empty_figure("This run has no crack dumps", scheme)
        if not num:
            return empty_figure("Select a crack dump", scheme)
        elems = ds.crack_dump(int(num))
        if not elems:
            return empty_figure("Crack dump could not be read (see Datasets → issues)", scheme)
        fig = go.Figure()
        state_names = {0: "stuck", 1: "slipping", 2: "open"}
        colors = {0: "#2f9e44", 1: "#1c7ed6", 2: "#e03131"}
        for e in elems:
            st = e.state()
            pos = np.arange(len(st))
            for code, nm in state_names.items():
                m = st == code
                if m.any():
                    fig.add_trace(go.Bar(x=pos[m], y=np.ones(m.sum()), marker_color=colors[code],
                                         name=f"crack {e.stope_id}: {nm}"))
            for k, fld in enumerate(("T1slp", "Taslp")):
                if fld in e.fields:
                    fig.add_trace(go.Scatter(x=pos, y=e.field(fld), mode="lines", yaxis="y2", name=fld,
                                             line=dict(color=PALETTE[k + 4])))
        fig.update_layout(template=template(scheme), barmode="overlay", yaxis=dict(visible=False),
                          yaxis2=dict(overlaying="y", side="right", title="slip"), xaxis_title="element")
        return fig

    @app.callback(Output("geo-info", "children"), Input("store-ds", "data"))
    def _info(ds_id):
        ds = state.get_dataset(ds_id)
        geom = ds.geometry if ds else None
        if geom is None:
            return dmc.Text("No geometry in this dataset.", c="dimmed")
        parts = [section("Grid"), kv_table([
            ("Nodes", f"i {geom.grid.i1}–{geom.grid.i2}, j {geom.grid.j1}–{geom.grid.j2}, k {geom.grid.k1}–"
                      f"{geom.grid.k2}" if geom.grid else "—"),
            ("Spacing", " / ".join(f"{geom.spacing[a]:g} m" for a in "ijk")),
            ("Size", " × ".join(f"{v:g} m" for v in geom.physical_size) if geom.physical_size else "—"),
            ("Source centroid (grid)", "({}, {}, {})".format(*geom.source_centroid()) if geom.sources else "—"),
        ])]
        if geom.props:
            parts += [section("Material properties"), simple_table(
                ["#", "K (Pa)", "G (Pa)", "ρ (kg/m³)", "vp (m/s)", "vs (m/s)", "ν"],
                [[p.number, f"{p.bulk:.4g}", f"{p.shear:.4g}", f"{p.density:.4g}", f"{p.vp:.1f}", f"{p.vs:.1f}",
                  f"{p.poisson:.3f}"] for p in geom.props], fz="sm")]
        if geom.materials:
            parts += [section("Material zones"), simple_table(
                ["material", "i", "j", "k"], [[m.number, f"{m.extent.i1}–{m.extent.i2}", f"{m.extent.j1}–{m.extent.j2}",
                                               f"{m.extent.k1}–{m.extent.k2}"] for m in geom.materials], fz="sm")]
        if geom.sources:
            parts += [section("Sources"), simple_table(
                ["component", "i", "j", "k", "amplitude", "period (s)"],
                [[s.component, f"{s.extent.i1}–{s.extent.i2}", f"{s.extent.j1}–{s.extent.j2}",
                  f"{s.extent.k1}–{s.extent.k2}", f"{s.params.get('amplitude', 0):.3g}",
                  f"{s.params.get('period', 0):.3g}"] for s in geom.sources], fz="sm")]
        if geom.stopes:
            parts += [section("Stopes / faults"), simple_table(
                ["id", "type", "i", "j", "k", "kn", "ks", "friction (°)", "cohesion (Pa)"],
                [[s.stope_id, s.stype, f"{s.extent.i1}–{s.extent.i2}", f"{s.extent.j1}–{s.extent.j2}",
                  f"{s.extent.k1}–{s.extent.k2}", f"{s.params.get('kn', 0):.3g}", f"{s.params.get('ks', 0):.3g}",
                  f"{s.params.get('friction_deg', 0):.3g}", f"{s.params.get('cohesion', 0):.3g}"]
                 for s in geom.stopes], fz="sm")]
        if geom.boundary_stress:
            parts += [section("Boundary / initial stresses"), kv_table([(k, f"{v:.4g} Pa")
                                                                       for k, v in geom.boundary_stress.items()])]
        return dmc.Stack(parts, gap=4)
