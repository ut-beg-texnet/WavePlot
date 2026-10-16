"""Snapshots page: animated 2-D field planes with colour maps, contours, multi-line views,
geometry overlays, multi-panel grids, a probe inspector and exports."""

from __future__ import annotations

import time
from urllib.parse import parse_qs, quote

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update

from ...render.panels import snapshot_panel
from ...render.snapshot_fig import SnapshotStyle, snapshot_figure
from ...render.theme import empty_figure, si_format, template
from ...science.colormaps import CMAPS, DEFAULT_DIVERGING, DEFAULT_SEQUENTIAL
from ...science.levels import METHODS
from .. import state
from ..components import labeled, provenance_badge, section

REPS = [{"label": "Map", "value": "map"}, {"label": "Contours", "value": "contour"},
        {"label": "Lines", "value": "lines"}, {"label": "Map+contours", "value": "map+contour"}]
RANGES = [{"label": "Robust", "value": "robust"}, {"label": "Frame", "value": "frame"},
          {"label": "Series", "value": "series"}, {"label": "Fixed", "value": "fixed"}]
OVERLAYS = [("materials", "Materials"), ("sources", "Sources"), ("stopes", "Stopes / faults"),
            ("receivers", "Receivers")]
SPEEDS = [{"label": f"{v} fps", "value": str(v)} for v in (1, 2, 4, 8, 12)]
METHOD_CYCLE = ["lin", "quad", "dquad", "cos", "fixed", "log"]


def _cmap_options():
    groups: dict[str, list] = {}
    for c in CMAPS.values():
        groups.setdefault(c.kind.capitalize(), []).append({"value": c.name, "label": c.label})
    return [{"group": k, "items": v} for k, v in groups.items()]


def layout(series: str | None = None, **_kwargs):
    controls = dmc.Stack([
        section("Data"),
        dmc.Select(id="snp-series", data=[], value=series, searchable=True, placeholder="Snapshot series",
                   label="Series", comboboxProps={"withinPortal": True}),
        section("Display"),
        dmc.SegmentedControl(id="snp-rep", data=REPS, value="map", fullWidth=True),
        dmc.Group([
            dmc.Select(id="snp-cmap", data=_cmap_options(), value=DEFAULT_DIVERGING, label="Colormap", w=190,
                       comboboxProps={"withinPortal": True}),
            dmc.Switch(id="snp-reverse", label="Reverse", mt=20),
        ], gap="xs", align="flex-start"),
        labeled("Colour range", dmc.SegmentedControl(id="snp-range", data=RANGES, value="robust", fullWidth=True),
                "Robust: 0.5–99.5 percentile of this frame (ignores isolated extreme cells). Frame: full range of "
                "this frame (as v407). Series: fixed over all frames (from the .map header). Fixed: your values "
                "(press K to lock the current range)."),
        dmc.Group([
            dmc.NumberInput(id="snp-vmin", label="Min", w=118, decimalScale=6, disabled=True),
            dmc.NumberInput(id="snp-vmax", label="Max", w=118, decimalScale=6, disabled=True),
        ], gap="xs", grow=True),
        dmc.Group([
            dmc.Switch(id="snp-symmetric", label="Symmetric about 0", checked=True),
            dmc.Switch(id="snp-abs", label="|value|", checked=False),
            dmc.Switch(id="snp-smooth", label="Smooth", checked=False),
        ], gap="sm"),
        section("Levels"),
        dmc.Group([
            dmc.Switch(id="snp-banded", label="Banded colours", checked=False),
        ]),
        dmc.Group([
            dmc.Select(id="snp-method", data=[{"value": k, "label": v} for k, v in METHODS.items()], value="lin",
                       label="Distribution", w=170, comboboxProps={"withinPortal": True}),
            dmc.NumberInput(id="snp-nlev", label="Levels", value=11, min=2, max=256, w=80),
        ], gap="xs"),
        section("Multi-line view"),
        dmc.Group([
            dmc.SegmentedControl(id="snp-linedir", data=[{"label": "Rows", "value": "h"},
                                                         {"label": "Columns", "value": "v"},
                                                         {"label": "Both", "value": "both"}], value="h"),
            dmc.NumberInput(id="snp-linescale", label=None, value=1.0, min=0.05, step=0.25, decimalScale=2, w=80,
                            leftSection="×"),
        ], gap="xs"),
        section("Overlays"),
        dmc.CheckboxGroup(id="snp-overlays", value=[k for k, _ in OVERLAYS],
                          children=dmc.SimpleGrid([dmc.Checkbox(label=lab, value=k) for k, lab in OVERLAYS], cols=2,
                                                  spacing=4, verticalSpacing=4)),
        dmc.Switch(id="snp-fillmat", label="Fill materials", checked=False),
        section("Panels"),
        dmc.Group([
            dmc.NumberInput(id="snp-prows", label="Rows", value=1, min=1, max=6, w=70),
            dmc.NumberInput(id="snp-pcols", label="Cols", value=1, min=1, max=6, w=70),
            dmc.NumberInput(id="snp-pstep", label="Frame step", value=1, min=1, w=90),
        ], gap="xs"),
        dmc.SegmentedControl(id="snp-pmode", data=[{"label": "Successive frames", "value": "frames"},
                                                   {"label": "Several series", "value": "series"}],
                             value="frames", fullWidth=True),
        dmc.MultiSelect(id="snp-pseries", data=[], placeholder="Series for the panels", searchable=True,
                        comboboxProps={"withinPortal": True}),
        dmc.Switch(id="snp-plinked", label="Link colour scales", checked=True),
        section("Export"),
        dmc.Group([
            dmc.Button("PNG", id="snp-dl-png", size="xs", variant="default"),
            dmc.Button("PDF", id="snp-dl-pdf", size="xs", variant="default"),
            dmc.Button("CSV", id="snp-dl-csv", size="xs", variant="default"),
            dmc.Button("Envelope", id="snp-envelope", size="xs", variant="default"),
        ], gap=6),
        dmc.Group([
            dmc.Button("Animation (MP4)", id="snp-dl-mp4", size="xs", variant="light"),
            dmc.Button("GIF", id="snp-dl-gif", size="xs", variant="light"),
        ], gap=6),
        dcc.Download(id="snp-download"),
        html.Div(id="snp-export-status"),
    ], gap=6)

    frame_bar = dmc.Group([
        dmc.ActionIconGroup([
            dmc.ActionIcon("⏮", id="snp-first", variant="default", size="lg"),
            dmc.ActionIcon("◀", id="snp-prev", variant="default", size="lg"),
            dmc.ActionIcon("▶", id="snp-play", variant="filled", size="lg"),
            dmc.ActionIcon("▶", id="snp-next", variant="default", size="lg", className="wp-next"),
            dmc.ActionIcon("⏭", id="snp-last", variant="default", size="lg"),
        ]),
        dmc.Slider(id="snp-frame", min=0, max=0, value=0, step=1, label=None, style={"flex": 1}, size="sm"),
        dmc.Text(id="snp-time", className="wp-mono", w=150, ta="right"),
        dmc.NumberInput(id="snp-jump", value=1, min=1, w=70, leftSection="±", size="xs"),
        dmc.Select(id="snp-speed", data=SPEEDS, value="4", w=90, size="xs", comboboxProps={"withinPortal": True}),
    ], className="wp-frame-bar", wrap="nowrap", mt=4)

    inspector = dmc.Paper([
        section("Inspector", dmc.Text("click the plot to probe a cell", size="xs", c="dimmed")),
        dmc.Grid([
            dmc.GridCol(html.Div(id="snp-probe-info", className="wp-mono"), span={"base": 12, "md": 4}),
            dmc.GridCol(dcc.Graph(id="snp-probe-graph", figure=empty_figure("Click on the plot to see the time "
                                                                              "series of a cell"),
                                  style={"height": 220}, config={"displaylogo": False}), span={"base": 12, "md": 8}),
        ]),
    ], mt="sm")

    return dmc.Grid([
        dmc.GridCol(dmc.Paper(controls, className="wp-controls"), span={"base": 12, "md": 3}),
        dmc.GridCol([
            dmc.Paper([
                dmc.Group([dmc.Text(id="snp-title", fw=600, size="sm"), html.Div(id="snp-badge")],
                          justify="space-between"),
                dcc.Loading(dcc.Graph(id="snp-graph", className="wp-graph",
                                      config={"displaylogo": False, "scrollZoom": True,
                                              "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                                              "toImageButtonOptions": {"format": "png", "scale": 2,
                                                                       "filename": "waveplot_snapshot"}}),
                            delay_show=400, type="dot"),
                frame_bar,
            ]),
            inspector,
            dcc.Interval(id="snp-interval", interval=250, disabled=True),
            dcc.Store(id="snp-probe"),
        ], span={"base": 12, "md": 9}),
    ], gutter="sm")


def _style(rep, cmap, reverse, rng, vmin, vmax, symmetric, abs_, smooth, banded, method, nlev, linedir,
           linescale, overlays, fillmat, prefs) -> SnapshotStyle:
    return SnapshotStyle(
        representation=rep or "map", cmap=cmap or DEFAULT_DIVERGING, reverse=bool(reverse),
        range_mode=rng or "robust", vmin=vmin if isinstance(vmin, int | float) else None,
        vmax=vmax if isinstance(vmax, int | float) else None, symmetric=bool(symmetric),
        abs_values=bool(abs_), smooth=bool(smooth), banded=bool(banded), level_method=method or "lin",
        n_levels=int(nlev or 11), line_dir=linedir or "h", line_scale=float(linescale or 1.0),
        overlays=list(overlays or []), fill_materials=bool(fillmat), mode=state.mode_of(prefs),
        scheme=state.scheme_of(prefs))


def register_callbacks(app) -> None:
    @app.callback(
        Output("snp-series", "data"), Output("snp-series", "value"), Output("snp-pseries", "data"),
        Input("store-ds", "data"), State("snp-series", "value"), State("url", "search"),
    )
    def _series_options(ds_id, current, search):
        ds = state.get_dataset(ds_id)
        if ds is None:
            return [], None, []
        groups: dict[str, list] = {}
        for s in ds.series.values():
            key = "Stope / fault planes" if s.stope_id else ("Space–time" if s.is_spacetime else
                                                             f"{s.axes} planes")
            groups.setdefault(key, []).append({"value": s.slug, "label": s.label})
        data = [{"group": k, "items": v} for k, v in groups.items()]
        q = parse_qs((search or "").lstrip("?"))
        want = (q.get("series") or [None])[0] or current
        if want not in ds.series:
            want = next((s.slug for s in ds.series.values() if not s.is_stope and len(s) > 1),
                        next(iter(ds.series), None))
        flat = [{"value": s.slug, "label": s.label} for s in ds.series.values()]
        return data, want, flat

    @app.callback(
        Output("snp-frame", "max"), Output("snp-frame", "marks"),
        Input("snp-series", "value"), State("store-ds", "data"),
    )
    def _frame_bounds(slug, ds_id):
        ds = state.get_dataset(ds_id)
        s = ds.series.get(slug) if ds and slug else None
        if s is None:
            return 0, []
        n = len(s)
        ticks = np.unique(np.linspace(0, n - 1, min(n, 6)).round().astype(int))
        return n - 1, [{"value": int(k), "label": f"{s.times[k]:.3g}s"} for k in ticks]

    # --- frame controller ---------------------------------------------------------------------
    @app.callback(
        Output("snp-frame", "value"),
        Input("snp-first", "n_clicks"), Input("snp-prev", "n_clicks"), Input("snp-next", "n_clicks"),
        Input("snp-last", "n_clicks"), Input("snp-interval", "n_intervals"), Input("store-kbd", "data"),
        Input("snp-series", "value"),
        State("snp-frame", "value"), State("snp-frame", "max"), State("snp-jump", "value"),
        prevent_initial_call=True,
    )
    def _frame(_f, _p, _n, _l, _tick, kbd, _series, value, vmax, jump):
        trig = callback_context.triggered_id
        value, vmax, jump = int(value or 0), int(vmax or 0), max(int(jump or 1), 1)
        if trig == "snp-series":
            return min(value, vmax)
        if trig == "store-kbd":
            if not kbd or kbd.get("path") != "/snapshots":
                return no_update
            key = kbd.get("key")
            step = jump * (5 if kbd.get("shift") else 1)
            if key == "ArrowRight":
                return min(value + step, vmax)
            if key == "ArrowLeft":
                return max(value - step, 0)
            if key == "Home":
                return 0
            if key == "End":
                return vmax
            return no_update
        if trig == "snp-first":
            return 0
        if trig == "snp-last":
            return vmax
        if trig == "snp-prev":
            return max(value - jump, 0)
        if trig == "snp-next":
            return min(value + jump, vmax)
        if trig == "snp-interval":
            return (value + jump) % (vmax + 1) if vmax > 0 else 0
        return no_update

    @app.callback(
        Output("snp-interval", "disabled"), Output("snp-interval", "interval"), Output("snp-play", "children"),
        Output("snp-speed", "value"),
        Input("snp-play", "n_clicks"), Input("store-kbd", "data"), Input("snp-speed", "value"),
        State("snp-interval", "disabled"),
        prevent_initial_call=True,
    )
    def _play(_n, kbd, speed, disabled):
        trig = callback_context.triggered_id
        speeds = [s["value"] for s in SPEEDS]
        speed = speed if speed in speeds else "4"
        if trig == "store-kbd":
            if not kbd or kbd.get("path") != "/snapshots":
                return no_update, no_update, no_update, no_update
            key = kbd.get("key")
            if key == " ":
                disabled = not disabled
            elif key in ("+", "="):
                speed = speeds[min(speeds.index(speed) + 1, len(speeds) - 1)]
            elif key in ("-", "_"):
                speed = speeds[max(speeds.index(speed) - 1, 0)]
            else:
                return no_update, no_update, no_update, no_update
        elif trig == "snp-play":
            disabled = not disabled
        return disabled, int(1000 / float(speed)), ("▶" if disabled else "⏸"), speed

    # --- keyboard display toggles -------------------------------------------------------------
    @app.callback(
        Output("snp-rep", "value"), Output("snp-abs", "checked"), Output("snp-banded", "checked"),
        Output("snp-nlev", "value"), Output("snp-method", "value"), Output("snp-smooth", "checked"),
        Output("snp-overlays", "value"), Output("snp-linedir", "value"), Output("snp-linescale", "value"),
        Output("snp-range", "value"), Output("snp-vmin", "value"), Output("snp-vmax", "value"),
        Input("store-kbd", "data"),
        State("snp-rep", "value"), State("snp-abs", "checked"), State("snp-banded", "checked"),
        State("snp-nlev", "value"), State("snp-method", "value"), State("snp-smooth", "checked"),
        State("snp-overlays", "value"), State("snp-linescale", "value"), State("snp-graph", "figure"),
        prevent_initial_call=True,
    )
    def _keys(kbd, rep, abs_, banded, nlev, method, smooth, overlays, linescale, fig):
        out = [no_update] * 12
        if not kbd or kbd.get("path") != "/snapshots":
            return out
        k = kbd.get("key")
        if k in ("m", "M"):
            out[0] = "map"
        elif k in ("c", "C"):
            out[0] = "contour"
        elif k in ("l", "L"):
            out[0] = "lines"
        elif k in ("a", "A"):
            out[1] = not abs_
        elif k in ("b", "B"):
            out[2] = not banded
        elif k == ">":
            out[3] = min(int(nlev or 11) + 2, 256)
        elif k == "<":
            out[3] = max(int(nlev or 11) - 2, 2)
        elif k in ("d", "D"):
            i = METHOD_CYCLE.index(method) if method in METHOD_CYCLE else 0
            out[4] = METHOD_CYCLE[(i + 1) % len(METHOD_CYCLE)]
        elif k in ("i", "I"):
            out[5] = not smooth
        elif k in ("g", "G"):
            out[6] = [] if overlays else [o for o, _ in OVERLAYS]
        elif k in ("h", "H"):
            out[7] = "h"
        elif k in ("v", "V"):
            out[7] = "v"
        elif k == "X":
            out[8] = round(float(linescale or 1) * 1.25, 3)
        elif k == "x":
            out[8] = round(float(linescale or 1) * 0.8, 3)
        elif k in ("k", "K"):
            ca = (fig or {}).get("layout", {}).get("coloraxis", {})
            if "cmin" in ca and "cmax" in ca:
                out[9], out[10], out[11] = "fixed", ca["cmin"], ca["cmax"]
        return out

    @app.callback(Output("snp-vmin", "disabled"), Output("snp-vmax", "disabled"), Input("snp-range", "value"))
    def _fixed_inputs(rng):
        return rng != "fixed", rng != "fixed"

    @app.callback(Output("snp-cmap", "value"), Input("snp-abs", "checked"), Input("snp-series", "value"),
                  State("snp-cmap", "value"), State("store-ds", "data"), prevent_initial_call=True)
    def _auto_cmap(abs_, slug, cmap, ds_id):
        """Sequential colormap for non-negative data (|value|, magnitudes), diverging for signed data."""
        ds = state.get_dataset(ds_id)
        s = ds.series.get(slug) if ds and slug else None
        nonneg = bool(abs_) or (s is not None and s.global_range()[0] >= 0)
        kind = CMAPS[cmap].kind if cmap in CMAPS else ""
        if nonneg and kind == "diverging":
            return DEFAULT_SEQUENTIAL
        if not nonneg and kind == "sequential":
            return DEFAULT_DIVERGING
        return no_update

    # --- rendering -----------------------------------------------------------------------------
    style_inputs = [Input("snp-rep", "value"), Input("snp-cmap", "value"), Input("snp-reverse", "checked"),
                    Input("snp-range", "value"), Input("snp-vmin", "value"), Input("snp-vmax", "value"),
                    Input("snp-symmetric", "checked"), Input("snp-abs", "checked"), Input("snp-smooth", "checked"),
                    Input("snp-banded", "checked"), Input("snp-method", "value"), Input("snp-nlev", "value"),
                    Input("snp-linedir", "value"), Input("snp-linescale", "value"), Input("snp-overlays", "value"),
                    Input("snp-fillmat", "checked"), Input("store-prefs", "data")]

    @app.callback(
        Output("snp-graph", "figure"), Output("snp-title", "children"), Output("snp-badge", "children"),
        Output("snp-time", "children"),
        Input("store-ds", "data"), Input("snp-series", "value"), Input("snp-frame", "value"),
        Input("snp-prows", "value"), Input("snp-pcols", "value"), Input("snp-pstep", "value"),
        Input("snp-pmode", "value"), Input("snp-pseries", "value"), Input("snp-plinked", "checked"),
        *style_inputs,
    )
    def _render(ds_id, slug, frame, prows, pcols, pstep, pmode, pseries, plinked, *style_args):

        style = _style(*style_args)
        ds = state.get_dataset(ds_id)
        if ds is None:
            return empty_figure("Select a dataset", style.scheme), "", None, ""
        s = ds.series.get(slug) if slug else None
        if s is None:
            return empty_figure("This dataset has no snapshots" if not ds.series else "Select a series",
                                style.scheme), "", None, ""
        n = int(np.clip(frame or 0, 0, len(s) - 1))
        rows, cols = int(prows or 1), int(pcols or 1)
        panels = []
        try:
            if rows * cols == 1:
                panels = [snapshot_panel(ds, s, n)]
            elif pmode == "series" and pseries:
                t = s.records[n].time
                for slug2 in pseries[:rows * cols]:
                    s2 = ds.series.get(slug2)
                    if s2 is not None:
                        k = s2.frame_index_for_time(t)
                        r = s2.records[k]
                        panels.append(snapshot_panel(ds, s2, k, title=f"{r.name} [{r.axes}] t={r.time:.4g}s"))
                rows = max(1, -(-len(panels) // cols))
            else:
                step = max(int(pstep or 1), 1)
                for p in range(rows * cols):
                    k = n + p * step
                    if k >= len(s):
                        break
                    panels.append(snapshot_panel(ds, s, k, title=f"t = {s.records[k].time:.4g} s"))
        except Exception as exc:
            return empty_figure(f"Cannot read data: {exc}", style.scheme), "", None, ""
        fig, metas = snapshot_figure(panels, style, geometry=ds.geometry, receivers=ds.receivers, rows=rows,
                                     cols=cols, linked=bool(plinked), uirevision=f"{ds_id}|{slug}|{rows}x{cols}")
        r = s.records[n]
        title = f"{s.long_name} — {r.name} on {r.axes} plane" + (f" at {r.fixed_axis} = {r.fixed_index}"
                                                                  if r.fixed_axis else "")
        quirks = sorted({q for m in metas for q in m["provenance"].quirks})
        lo, hi = metas[0]["range"]
        tlabel = f"t = {r.time:.5g} s  ({n + 1}/{len(s)})"
        return fig, [title, dmc.Text(f"colour range {si_format(lo, s.units)} … {si_format(hi, s.units)}",
                                     size="xs", c="dimmed", span=True, ml=8)], \
            provenance_badge(quirks, style.mode), tlabel

    # --- inspector -----------------------------------------------------------------------------
    @app.callback(
        Output("snp-probe-info", "children"), Output("snp-probe-graph", "figure"),
        Output("url", "pathname", allow_duplicate=True), Output("store-nav", "data", allow_duplicate=True),
        Input("snp-graph", "clickData"), Input("snp-frame", "value"),
        State("store-ds", "data"), State("snp-series", "value"), State("store-prefs", "data"),
        prevent_initial_call=True,
    )
    def _probe(click, frame, ds_id, slug, prefs):
        scheme = state.scheme_of(prefs)
        ds = state.get_dataset(ds_id)
        s = ds.series.get(slug) if ds and slug else None
        if s is None or not click or not click.get("points"):
            return no_update, no_update, no_update, no_update
        pt = click["points"][0]
        cd = pt.get("customdata")
        if callback_context.triggered_id == "snp-graph" and cd and isinstance(cd, list) and len(cd) > 1:
            hist = [int(h) for h in cd[1:] if h is not None]
            return no_update, no_update, "/histories", {"open_histories": hist, "ds": ds_id, "ts": time.time()}
        a1, a2 = s.coords()
        ix, iy = a1.nearest(float(pt["x"])), a2.nearest(float(pt["y"]))
        n = int(np.clip(frame or 0, 0, len(s) - 1))
        try:
            series_vals = s.cell_series(ix, iy)
        except Exception as exc:
            return dmc.Alert(str(exc), color="red"), no_update, no_update, no_update
        val = float(series_vals[n])
        rec = s.records[n]
        ijk = {rec.fixed_axis: rec.fixed_index} if rec.fixed_axis else {}
        ijk[a1.axis] = int(a1.index[ix])
        ijk[a2.axis] = int(a2.index[iy])
        mat = ""
        geom = ds.geometry
        if geom is not None and all(a in ijk for a in "ijk"):
            for m in reversed(geom.materials):
                if all(m.extent.range(a)[0] <= ijk[a] <= m.extent.range(a)[1] for a in "ijk"):
                    p = geom.prop_for(m.number)
                    mat = f"material {m.number}" + (f" (vp {p.vp:.0f}, vs {p.vs:.0f} m/s)" if p else "")
                    break
        info = dmc.Stack([
            dmc.Text(f"{s.var_name} = {si_format(val, s.units, 6)}", fw=700),
            dmc.Text("grid  " + "  ".join(f"{a}={ijk[a]}" for a in "ijk" if a in ijk), size="xs"),
            dmc.Text(f"{a1.label} = {a1.coord[ix]:.4g} {a1.unit},  {a2.label} = {a2.coord[iy]:.4g} {a2.unit}",
                     size="xs"),
            dmc.Text(f"t = {rec.time:.5g} s", size="xs"),
            dmc.Text(mat, size="xs", c="dimmed") if mat else None,
            dmc.Text(f"over all frames: min {si_format(float(series_vals.min()), s.units)}, "
                     f"max {si_format(float(series_vals.max()), s.units)}", size="xs", c="dimmed"),
        ], gap=2)
        fig = go.Figure(go.Scatter(x=s.times, y=series_vals, mode="lines+markers", marker=dict(size=4),
                                   name=s.var_name, hovertemplate="t=%{x:.4g} s<br>%{y:.4g}<extra></extra>"))
        fig.add_vline(x=float(rec.time), line=dict(color="#e8590c", width=1, dash="dot"))
        fig.update_layout(template=template(scheme), margin=dict(l=55, r=10, t=10, b=35),
                          xaxis_title="time (s)", yaxis_title=f"{s.var_name} ({s.units})" if s.units else s.var_name,
                          showlegend=False)
        return info, fig, no_update, no_update

    # --- exports ---------------------------------------------------------------------------------
    @app.callback(
        Output("snp-download", "data"), Output("snp-export-status", "children"),
        Input("snp-dl-png", "n_clicks"), Input("snp-dl-pdf", "n_clicks"), Input("snp-dl-csv", "n_clicks"),
        Input("snp-envelope", "n_clicks"),
        State("store-ds", "data"), State("snp-series", "value"), State("snp-frame", "value"),
        *[State(i.component_id, i.component_property) for i in style_inputs],
        prevent_initial_call=True,
    )
    def _export(_png, _pdf, _csv, _env, ds_id, slug, frame, *style_args):
        from ...export.images import render_panel_bytes
        from ...export.tables import frame_csv

        ds = state.get_dataset(ds_id)
        s = ds.series.get(slug) if ds and slug else None
        if s is None:
            return no_update, no_update
        style = _style(*style_args).model_copy(update={"scheme": "light"})
        n = int(np.clip(frame or 0, 0, len(s) - 1))
        base = f"{ds.name}_{slug.replace('|', '_').replace('=', '')}_f{n + 1:03d}"
        trig = callback_context.triggered_id
        if trig == "snp-dl-csv":
            return dcc.send_string(frame_csv(s, n), f"{base}.csv"), None
        if trig == "snp-envelope":
            env = s.envelope("absmax")
            panel = snapshot_panel(ds, s, n, data=env, title=f"{s.name}: peak |value| over {len(s)} frames")
            data = render_panel_bytes(panel, style.model_copy(update={"range_mode": "frame"}), ds, "png")
            return dcc.send_bytes(data, f"{base}_envelope.png"), dmc.Text(
                "Envelope = value of largest magnitude at each cell over all frames.", size="xs", c="dimmed")
        fmt = "pdf" if trig == "snp-dl-pdf" else "png"
        data = render_panel_bytes(snapshot_panel(ds, s, n), style, ds, fmt)
        return dcc.send_bytes(data, f"{base}.{fmt}"), None

    @app.callback(
        Output("snp-download", "data", allow_duplicate=True),
        Output("snp-export-status", "children", allow_duplicate=True),
        Input("snp-dl-mp4", "n_clicks"), Input("snp-dl-gif", "n_clicks"),
        State("store-ds", "data"), State("snp-series", "value"), State("snp-speed", "value"),
        *[State(i.component_id, i.component_property) for i in style_inputs],
        prevent_initial_call=True,
        running=[(Output("snp-dl-mp4", "loading"), True, False), (Output("snp-dl-gif", "loading"), True, False)],
    )
    def _animation(_mp4, _gif, ds_id, slug, speed, *style_args):
        import tempfile
        from pathlib import Path

        from ...export.images import export_series

        ds = state.get_dataset(ds_id)
        s = ds.series.get(slug) if ds and slug else None
        if s is None:
            return no_update, no_update
        fmt = "gif" if callback_context.triggered_id == "snp-dl-gif" else "mp4"
        style = _style(*style_args).model_copy(update={"scheme": "light"})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"anim.{fmt}"
            try:
                export_series(ds, s, out, frames=list(range(len(s))), style=style, fps=float(speed or 4))
            except Exception as exc:
                return no_update, dmc.Alert(f"Animation export failed: {exc}", color="red")
            data = out.read_bytes()
        name = f"{ds.name}_{slug.replace('|', '_').replace('=', '')}.{fmt}"
        return dcc.send_bytes(data, name), dmc.Text(f"{len(s)} frames exported", size="xs", c="dimmed")

    _ = (ALL, quote)
