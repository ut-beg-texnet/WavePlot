"""Compare page: the same quantity from two runs side by side or as a difference map, and
history overlays across runs (successor of v407's multi-file panels)."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from scipy.interpolate import RegularGridInterpolator

from ...formula.evaluate import make_loader
from ...render.panels import snapshot_panel
from ...render.snapshot_fig import SnapshotStyle, snapshot_figure
from ...render.theme import PALETTE, empty_figure, template
from .. import state
from ..components import section


def layout(**_kwargs):
    controls = dmc.Stack([
        section("Runs"),
        dmc.Text("Run A is the dataset selected in the header.", size="xs", c="dimmed"),
        dmc.Select(id="cmp-b", label="Run B", data=[], searchable=True, comboboxProps={"withinPortal": True}),
        section("Snapshots"),
        dmc.Select(id="cmp-series", label="Series (run A)", data=[], comboboxProps={"withinPortal": True}),
        dmc.Select(id="cmp-series-b", label="Series (run B)", data=[], comboboxProps={"withinPortal": True}),
        dmc.SegmentedControl(id="cmp-mode", data=[{"label": "Side by side", "value": "side"},
                                                  {"label": "A − B", "value": "diff"},
                                                  {"label": "Relative (A−B)/max|A|", "value": "rel"}],
                             value="side", fullWidth=True),
        dmc.Text("Frames are matched by simulation time; run B is re-gridded onto run A's grid when the grids "
                 "differ.", size="xs", c="dimmed"),
        section("Histories"),
        dmc.Select(id="cmp-hist", label="History (run A)", data=[], searchable=True,
                   comboboxProps={"withinPortal": True}),
    ], gap=6)
    main = dmc.Stack([
        dmc.Paper([
            dmc.Group([dmc.Text(id="cmp-title", fw=600, size="sm")]),
            dcc.Graph(id="cmp-graph", className="wp-graph-short", config={"displaylogo": False}),
            dmc.Slider(id="cmp-frame", min=0, max=0, value=0, step=1, mt=6),
        ]),
        dmc.Paper(dcc.Graph(id="cmp-hgraph", style={"height": 300}, config={"displaylogo": False})),
    ], gap="sm")
    return dmc.Grid([
        dmc.GridCol(dmc.Paper(controls, className="wp-controls"), span={"base": 12, "md": 3}),
        dmc.GridCol(main, span={"base": 12, "md": 9}),
    ], gutter="sm")


def register_callbacks(app) -> None:
    @app.callback(Output("cmp-b", "data"), Output("cmp-b", "value"), Output("cmp-series", "data"),
                  Output("cmp-series", "value"), Output("cmp-hist", "data"), Input("store-ds", "data"),
                  State("cmp-b", "value"))
    def _opts(ds_id, current_b):
        ds = state.get_dataset(ds_id)
        entries = state.registry().list_datasets()
        b_opts = [{"value": e.id, "label": e.id} for e in entries if e.id != ds_id]
        b_val = current_b if current_b in [o["value"] for o in b_opts] else (b_opts[0]["value"] if b_opts else None)
        if ds is None:
            return b_opts, b_val, [], None, []
        s_opts = [{"value": s.slug, "label": s.label} for s in ds.series.values()]
        h_opts = [{"value": str(h.number), "label": f"h{h.number} {h.name} {h.position}"} for h in ds.histories]
        return b_opts, b_val, s_opts, (s_opts[0]["value"] if s_opts else None), h_opts

    @app.callback(Output("cmp-series-b", "data"), Output("cmp-series-b", "value"), Input("cmp-b", "value"),
                  Input("cmp-series", "value"))
    def _opts_b(b_id, slug_a):
        dsb = state.get_dataset(b_id)
        if dsb is None:
            return [], None
        opts = [{"value": s.slug, "label": s.label} for s in dsb.series.values()]
        val = slug_a if slug_a in dsb.series else (opts[0]["value"] if opts else None)
        return opts, val

    @app.callback(Output("cmp-frame", "max"), Input("cmp-series", "value"), State("store-ds", "data"))
    def _fmax(slug, ds_id):
        ds = state.get_dataset(ds_id)
        s = ds.series.get(slug) if ds and slug else None
        return (len(s) - 1) if s else 0

    @app.callback(Output("cmp-graph", "figure"), Output("cmp-title", "children"),
                  Input("store-ds", "data"), Input("cmp-b", "value"), Input("cmp-series", "value"),
                  Input("cmp-series-b", "value"), Input("cmp-mode", "value"), Input("cmp-frame", "value"),
                  Input("store-prefs", "data"))
    def _render(ds_id, b_id, slug_a, slug_b, mode, frame, prefs):
        scheme = state.scheme_of(prefs)
        dsa, dsb = state.get_dataset(ds_id), state.get_dataset(b_id)
        if dsa is None or dsb is None:
            return empty_figure("Select two datasets (there must be at least two under the data roots)", scheme), ""
        sa, sb = dsa.series.get(slug_a or ""), dsb.series.get(slug_b or "")
        if sa is None or sb is None:
            return empty_figure("Select a series in both runs", scheme), ""
        n = int(np.clip(frame or 0, 0, len(sa) - 1))
        t = sa.records[n].time
        k = sb.frame_index_for_time(t)
        pa = snapshot_panel(dsa, sa, n, title=f"A: {dsa.name} — t = {t:.4g} s")
        pb = snapshot_panel(dsb, sb, k, title=f"B: {dsb.name} — t = {sb.records[k].time:.4g} s")
        style = SnapshotStyle(scheme=scheme, mode=state.mode_of(prefs), show_title=False)
        if mode == "side":
            fig, _ = snapshot_figure([pa, pb], style, rows=1, cols=2, linked=True, uirevision="cmp-side")
            return fig, f"{sa.long_name}: A frame {n + 1}, B frame {k + 1} (nearest time)"
        zb = pb.z
        if pb.z.shape != pa.z.shape or not (np.allclose(pa.x, pb.x) and np.allclose(pa.y, pb.y)):
            interp = RegularGridInterpolator((pb.x, pb.y), pb.z.astype(float), bounds_error=False, fill_value=np.nan)
            X, Y = np.meshgrid(pa.x, pa.y, indexing="ij")
            zb = interp(np.stack([X, Y], axis=-1))
        diff = pa.z.astype(float) - zb
        units = sa.units
        if mode == "rel":
            diff = diff / (np.nanmax(np.abs(pa.z)) or 1.0)
            units = "fraction of max|A|"
        pd = pa
        pd.z, pd.units, pd.title, pd.series_range = diff, units, "A − B", None
        fig, _ = snapshot_figure([pd], style.model_copy(update={"cmap": "balance", "symmetric": True}),
                                 geometry=dsa.geometry, receivers=dsa.receivers, uirevision="cmp-diff")
        rms = float(np.sqrt(np.nanmean(diff**2)))
        return fig, html.Span(f"A − B for {sa.long_name} at t = {t:.4g} s · RMS difference {rms:.4g} {units}")

    @app.callback(Output("cmp-hgraph", "figure"), Input("cmp-hist", "value"), Input("cmp-b", "value"),
                  Input("store-prefs", "data"), State("store-ds", "data"))
    def _hist(hnum, b_id, prefs, ds_id):
        scheme = state.scheme_of(prefs)
        dsa, dsb = state.get_dataset(ds_id), state.get_dataset(b_id)
        if dsa is None or not hnum:
            return empty_figure("Select a history to compare", scheme)
        ha = dsa.history_record(int(hnum))
        fig = go.Figure()
        ta = make_loader(dsa, "standard")(ha.number)
        fig.add_trace(go.Scatter(x=ta.x, y=ta.y, name=f"A: {dsa.name} h{ha.number}", line=dict(color=PALETTE[0])))
        if dsb is not None:
            match = next((h for h in dsb.histories if h.name == ha.name and h.position == ha.position), None)
            if match is not None:
                tb = make_loader(dsb, "standard")(match.number)
                fig.add_trace(go.Scatter(x=tb.x, y=tb.y, name=f"B: {dsb.name} h{match.number}",
                                         line=dict(color=PALETTE[1])))
            else:
                fig.add_annotation(text=f"run B has no {ha.name} history at {ha.position}", xref="paper",
                                   yref="paper", x=0.5, y=0.95, showarrow=False)
        fig.update_layout(template=template(scheme), xaxis_title="time (s)", yaxis_title=ha.units,
                          margin=dict(l=60, r=10, t=30, b=40), legend=dict(orientation="h", y=1.1),
                          title=dict(text=f"{ha.name} at {ha.position}", font=dict(size=12), x=0.01))
        return fig
