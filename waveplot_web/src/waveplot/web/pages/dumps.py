"""Dumps page: stress tensors and velocities on a plane — principal-stress crosses, velocity
arrows, derived scalar maps (ESS, isochromatics, principal / mean / von Mises stresses, …),
difference maps against a reference dump and a Mohr-circle inspector."""

from __future__ import annotations

from urllib.parse import parse_qs

import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, State, callback_context, dcc, html, no_update

from ...render.dump_fig import mohr_figure, principal_glyphs, velocity_arrows
from ...render.overlays import PlaneSpec
from ...render.snapshot_fig import PanelData, SnapshotStyle, snapshot_figure
from ...render.theme import empty_figure, si_format
from ...science import stress
from ...science.colormaps import CMAPS
from ...science.dumpfields import FIELDS, compute_field
from .. import state
from ..components import labeled, provenance_badge, section

KINDS = [{"label": "Principal axes", "value": "axes"}, {"label": "Velocity", "value": "vel"},
         {"label": "Scalar map", "value": "map"}]


def layout(plane: str | None = None, **_kwargs):
    cmaps = [{"value": c.name, "label": c.label} for c in CMAPS.values()]
    controls = dmc.Stack([
        section("Data"),
        dmc.Select(id="dmp-plane", data=[], value=plane, label="Plane", comboboxProps={"withinPortal": True}),
        section("Display"),
        dmc.SegmentedControl(id="dmp-kind", data=KINDS, value="map", fullWidth=True),
        dmc.Select(id="dmp-field", label="Scalar field", value="ess", searchable=True,
                   data=[{"value": k, "label": f.label} for k, f in FIELDS.items()],
                   comboboxProps={"withinPortal": True}),
        dmc.Group([
            dmc.NumberInput(id="dmp-theta", label="Plane θ (°)", value=0, step=5, w=95),
            dmc.NumberInput(id="dmp-phi", label="Friction φ (°)", value=30, min=0, max=89, step=1, w=95),
            dmc.NumberInput(id="dmp-coh", label="Cohesion (Pa)", value=0, step=1e5, w=110),
        ], gap="xs"),
        dmc.Text("ESS = |τ| − tanφ·|σn| − c on the plane whose normal is rotated θ from the vertical axis "
                 "(compression reduces the available shear).", size="xs", c="dimmed"),
        dmc.Select(id="dmp-ref", label="Difference from dump (Δ map)", data=[], clearable=True,
                   placeholder="none", comboboxProps={"withinPortal": True}),
        section("Glyphs"),
        dmc.Switch(id="dmp-glyphs", label="Overlay glyphs on the map", checked=False),
        dmc.Group([
            dmc.NumberInput(id="dmp-dens", label="Glyphs per axis", value=41, min=5, max=151, w=110),
            dmc.NumberInput(id="dmp-gscale", label="Length ×", value=1.0, min=0.1, step=0.25, decimalScale=2, w=90),
        ], gap="xs"),
        section("Colours"),
        dmc.Select(id="dmp-cmap", data=cmaps, value="balance", comboboxProps={"withinPortal": True}),
        labeled("Range", dmc.SegmentedControl(id="dmp-range", data=[{"label": "Robust", "value": "robust"},
                                                                   {"label": "Frame", "value": "frame"},
                                                                   {"label": "Fixed", "value": "fixed"}],
                                              value="robust", fullWidth=True)),
        dmc.Group([dmc.NumberInput(id="dmp-vmin", placeholder="min", w=118, disabled=True, decimalScale=6),
                   dmc.NumberInput(id="dmp-vmax", placeholder="max", w=118, disabled=True, decimalScale=6)],
                  gap="xs", grow=True),
        dmc.Group([dmc.Switch(id="dmp-banded", label="Banded", checked=False),
                   dmc.Switch(id="dmp-smooth", label="Smooth", checked=False)]),
        section("Overlays"),
        dmc.CheckboxGroup(id="dmp-overlays", value=["materials", "sources", "stopes"], children=dmc.Group([
            dmc.Checkbox(label="Materials", value="materials"), dmc.Checkbox(label="Sources", value="sources"),
            dmc.Checkbox(label="Stopes", value="stopes"), dmc.Checkbox(label="Receivers", value="receivers")])),
    ], gap=6)
    frame_bar = dmc.Group([
        dmc.ActionIconGroup([
            dmc.ActionIcon("⏮", id="dmp-first", variant="default", size="lg"),
            dmc.ActionIcon("◀", id="dmp-prev", variant="default", size="lg"),
            dmc.ActionIcon("▶", id="dmp-play", variant="filled", size="lg"),
            dmc.ActionIcon("▶", id="dmp-next", variant="default", size="lg"),
            dmc.ActionIcon("⏭", id="dmp-last", variant="default", size="lg"),
        ]),
        dmc.Slider(id="dmp-frame", min=0, max=0, value=0, step=1, style={"flex": 1}, size="sm"),
        dmc.Text(id="dmp-time", className="wp-mono", w=160, ta="right"),
    ], className="wp-frame-bar", wrap="nowrap", mt=4)
    inspector = dmc.Paper([
        section("Mohr circle", dmc.Text("click the plot to inspect the stress at a point", size="xs", c="dimmed")),
        dmc.Grid([
            dmc.GridCol(html.Div(id="dmp-probe-info", className="wp-mono"), span={"base": 12, "md": 5}),
            dmc.GridCol(dcc.Graph(id="dmp-mohr", figure=empty_figure("Click on the plot"), style={"height": 300},
                                  config={"displaylogo": False}), span={"base": 12, "md": 7}),
        ]),
    ], mt="sm")
    return dmc.Grid([
        dmc.GridCol(dmc.Paper(controls, className="wp-controls"), span={"base": 12, "md": 3}),
        dmc.GridCol([
            dmc.Paper([
                dmc.Group([dmc.Text(id="dmp-title", fw=600, size="sm"), html.Div(id="dmp-badge")],
                          justify="space-between"),
                dcc.Loading(dcc.Graph(id="dmp-graph", className="wp-graph",
                                      config={"displaylogo": False, "scrollZoom": True,
                                              "toImageButtonOptions": {"scale": 2, "filename": "waveplot_dump"}}),
                            delay_show=400, type="dot"),
                frame_bar,
            ]),
            inspector,
            dcc.Interval(id="dmp-interval", interval=500, disabled=True),
        ], span={"base": 12, "md": 9}),
    ], gutter="sm")


def _plane(ds, s) -> PlaneSpec:
    r = s.first
    sp = dict(ds.spacing)
    d1, d2 = r.spacing
    sp[r.ax1], sp[r.ax2] = d1 or sp[r.ax1], d2 or sp[r.ax2]
    return PlaneSpec(r.ax1, r.ax2, r.fixed_axis, r.fixed_index, sp)


def register_callbacks(app) -> None:
    @app.callback(Output("dmp-plane", "data"), Output("dmp-plane", "value"), Input("store-ds", "data"),
                  State("dmp-plane", "value"), State("url", "search"))
    def _planes(ds_id, current, search):
        ds = state.get_dataset(ds_id)
        if ds is None or not ds.dump_series:
            return [], None
        data = [{"value": s.slug, "label": s.label} for s in ds.dump_series.values()]
        q = parse_qs((search or "").lstrip("?"))
        want = (q.get("plane") or [None])[0] or current
        if want not in ds.dump_series:
            want = next(iter(ds.dump_series))
        return data, want

    @app.callback(Output("dmp-frame", "max"), Output("dmp-frame", "marks"), Output("dmp-ref", "data"),
                  Input("dmp-plane", "value"), State("store-ds", "data"))
    def _bounds(slug, ds_id):
        ds = state.get_dataset(ds_id)
        s = ds.dump_series.get(slug) if ds and slug else None
        if s is None:
            return 0, [], []
        n = len(s)
        ticks = np.unique(np.linspace(0, n - 1, min(n, 6)).round().astype(int))
        refs = [{"value": str(k), "label": f"dump #{r.number} (t = {r.time:.4g} s)"} for k, r in enumerate(s.records)]
        return n - 1, [{"value": int(k), "label": f"{s.times[k]:.3g}s"} for k in ticks], refs

    @app.callback(Output("dmp-frame", "value"),
                  Input("dmp-first", "n_clicks"), Input("dmp-prev", "n_clicks"), Input("dmp-next", "n_clicks"),
                  Input("dmp-last", "n_clicks"), Input("dmp-interval", "n_intervals"), Input("store-kbd", "data"),
                  State("dmp-frame", "value"), State("dmp-frame", "max"), prevent_initial_call=True)
    def _frame(_a, _b, _c, _d, _tick, kbd, value, vmax):
        trig = callback_context.triggered_id
        value, vmax = int(value or 0), int(vmax or 0)
        if trig == "store-kbd":
            if not kbd or kbd.get("path") != "/dumps":
                return no_update
            key = kbd.get("key")
            trig = {"ArrowRight": "dmp-next", "ArrowLeft": "dmp-prev", "Home": "dmp-first", "End": "dmp-last"}.get(key)
        return {"dmp-first": 0, "dmp-last": vmax, "dmp-prev": max(value - 1, 0), "dmp-next": min(value + 1, vmax),
                "dmp-interval": (value + 1) % (vmax + 1) if vmax else 0}.get(trig, no_update)

    @app.callback(Output("dmp-interval", "disabled"), Output("dmp-play", "children"),
                  Input("dmp-play", "n_clicks"), Input("store-kbd", "data"), State("dmp-interval", "disabled"),
                  prevent_initial_call=True)
    def _play(_n, kbd, disabled):
        if callback_context.triggered_id == "store-kbd":
            if not kbd or kbd.get("path") != "/dumps" or kbd.get("key") != " ":
                return no_update, no_update
        disabled = not disabled
        return disabled, ("▶" if disabled else "⏸")

    @app.callback(Output("dmp-vmin", "disabled"), Output("dmp-vmax", "disabled"), Output("dmp-field", "disabled"),
                  Input("dmp-range", "value"), Input("dmp-kind", "value"))
    def _enable(rng, kind):
        return rng != "fixed", rng != "fixed", kind != "map"

    @app.callback(
        Output("dmp-graph", "figure"), Output("dmp-title", "children"), Output("dmp-badge", "children"),
        Output("dmp-time", "children"),
        Input("store-ds", "data"), Input("dmp-plane", "value"), Input("dmp-frame", "value"), Input("dmp-kind", "value"),
        Input("dmp-field", "value"), Input("dmp-theta", "value"), Input("dmp-phi", "value"), Input("dmp-coh", "value"),
        Input("dmp-ref", "value"), Input("dmp-glyphs", "checked"), Input("dmp-dens", "value"),
        Input("dmp-gscale", "value"), Input("dmp-cmap", "value"), Input("dmp-range", "value"),
        Input("dmp-vmin", "value"), Input("dmp-vmax", "value"), Input("dmp-banded", "checked"),
        Input("dmp-smooth", "checked"), Input("dmp-overlays", "value"), Input("store-prefs", "data"),
    )
    def _render(ds_id, slug, frame, kind, field, theta, phi, coh, ref, glyphs, dens, gscale, cmap, rng, vmin, vmax,
                banded, smooth, overlays, prefs):
        scheme, mode = state.scheme_of(prefs), state.mode_of(prefs)
        ds = state.get_dataset(ds_id)
        if ds is None:
            return empty_figure("Select a dataset", scheme), "", None, ""
        s = ds.dump_series.get(slug) if slug else None
        if s is None:
            return empty_figure("This dataset has no dumps" if not ds.dump_series else "Select a plane", scheme), \
                "", None, ""
        n = int(np.clip(frame or 0, 0, len(s) - 1))
        a1, a2 = s.coords()
        try:
            comps = s.components(n)
            ip = s.in_plane(n)
        except Exception as exc:
            return empty_figure(f"Cannot read dump: {exc}", scheme), "", None, ""
        theta = float(theta or 0)
        phi = float(phi if phi is not None else 30)
        coh = float(coh or 0)
        rec = s.records[n]
        plane = _plane(ds, s)
        quirks: set[str] = set()
        style = SnapshotStyle(cmap=cmap or "balance", range_mode=rng or "robust",
                              vmin=vmin if isinstance(vmin, int | float) else None,
                              vmax=vmax if isinstance(vmax, int | float) else None, banded=bool(banded),
                              smooth=bool(smooth), overlays=list(overlays or []), mode=mode, scheme=scheme,
                              show_title=False)
        if kind == "map":
            spec = FIELDS.get(field or "ess", FIELDS["ess"])
            try:
                z = compute_field(spec.key, comps, ip, theta, phi, coh, mode)
                label = spec.label
                if ref not in (None, ""):
                    k = int(ref)
                    z = z - compute_field(spec.key, s.components(k), s.in_plane(k), theta, phi, coh, mode)
                    label = f"Δ {spec.label} (minus dump #{s.records[k].number})"
            except (ValueError, KeyError) as exc:
                return empty_figure(str(exc), scheme), "", None, ""
            if spec.key == "ess" and mode == "v407":
                quirks.add("ess_shear_sign")
            panel = PanelData(z=z, x=a1.coord, y=a2.coord, ax1=a1.axis, ax2=a2.axis, title="", units=spec.units,
                              label=spec.key, vertical_up=s.vertical_up, plane=plane)
            fig, metas = snapshot_figure([panel], style, geometry=ds.geometry, receivers=ds.receivers,
                                         uirevision=f"{ds_id}|{slug}")
            quirks |= set(metas[0]["provenance"].quirks)
            title = label
        else:
            zero = np.zeros(s.shape, np.float32)
            panel = PanelData(z=zero, x=a1.coord, y=a2.coord, ax1=a1.axis, ax2=a2.axis, title="", units="",
                              vertical_up=s.vertical_up, plane=plane)
            st2 = style.model_copy(update={"representation": "contour", "n_levels": 2})
            fig, _ = snapshot_figure([panel], st2, geometry=ds.geometry, receivers=ds.receivers,
                                     uirevision=f"{ds_id}|{slug}")
            fig.update_layout(coloraxis_showscale=False)
            title = "Principal stress axes (in-plane)" if kind == "axes" else "Velocity vectors (in-plane)"
        note = ""
        dens_v = int(dens or (51 if mode == "v407" else 41))
        if mode == "v407":
            quirks.add("dump_sampling")
        if kind == "axes" or (kind == "map" and glyphs):
            tr, note = principal_glyphs(a1.coord, a2.coord, ip["s11"], ip["s22"], ip["s12"], dens_v,
                                        float(gscale or 1), mode)
            for t in tr:
                fig.add_trace(t)
        elif kind == "vel":
            tr, note = velocity_arrows(a1.coord, a2.coord, ip["u"], ip["v"], dens_v, float(gscale or 1), mode)
            for t in tr:
                fig.add_trace(t)
        if note:
            fig.add_annotation(text=note, xref="paper", yref="paper", x=0, y=-0.1, showarrow=False, xanchor="left",
                               font=dict(size=11))
        fig.update_layout(showlegend=kind == "axes", legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))
        where = f"{rec.fixed_axis} = {rec.fixed_index}" if rec.fixed_axis else ""
        head = f"{title} — {rec.axes} plane {where} · {'3-D tensor' if rec.dim3 else '2-D'}"
        return fig, head, provenance_badge(sorted(quirks), mode), f"t = {rec.time:.5g} s ({n + 1}/{len(s)})"

    @app.callback(Output("dmp-probe-info", "children"), Output("dmp-mohr", "figure"),
                  Input("dmp-graph", "clickData"), Input("dmp-frame", "value"), Input("dmp-phi", "value"),
                  Input("dmp-coh", "value"), State("store-ds", "data"), State("dmp-plane", "value"),
                  State("store-prefs", "data"), prevent_initial_call=True)
    def _probe(click, frame, phi, coh, ds_id, slug, prefs):
        scheme = state.scheme_of(prefs)
        ds = state.get_dataset(ds_id)
        s = ds.dump_series.get(slug) if ds and slug else None
        if s is None or not click or not click.get("points"):
            return no_update, no_update
        pt = click["points"][0]
        if pt.get("x") is None or pt.get("y") is None:
            return no_update, no_update
        a1, a2 = s.coords()
        ix, iy = a1.nearest(float(pt["x"])), a2.nearest(float(pt["y"]))
        n = int(np.clip(frame or 0, 0, len(s) - 1))
        comps = {k: float(v[ix, iy]) for k, v in s.components(n).items()}
        ip = {k: float(v[ix, iy]) for k, v in s.in_plane(n).items()}
        if "s33" in comps:
            m = stress.mohr(ip["s11"], ip["s22"], ip["s12"], comps["s33"], comps["s23"], comps["s13"])
        else:
            m = stress.mohr(ip["s11"], ip["s22"], ip["s12"])
        rec = s.records[n]
        idx = {a1.axis: int(a1.index[ix]), a2.axis: int(a2.index[iy])}
        if rec.fixed_axis:
            idx[rec.fixed_axis] = rec.fixed_index
        lines = [dmc.Text("grid " + "  ".join(f"{a}={idx[a]}" for a in "ijk" if a in idx), size="xs", fw=700)]
        for k, v in comps.items():
            lines.append(dmc.Text(f"{k:>4} = {si_format(v, 'm/s' if k.startswith('v') else 'Pa', 5)}", size="xs"))
        lines.append(dmc.Text("principal: " + ", ".join(si_format(p, "Pa", 4) for p in m.principal), size="xs",
                              c="dimmed"))
        fig = mohr_figure(m, float(phi if phi is not None else 30), float(coh or 0), scheme,
                          f"t = {rec.time:.4g} s, " + ("3-D tensor" if "s33" in comps else "in-plane"))
        return dmc.Stack(lines, gap=0), fig
