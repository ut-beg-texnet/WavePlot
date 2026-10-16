"""Datasets page: browse runs under the data roots and inspect their contents."""

from __future__ import annotations

import datetime as _dt
from urllib.parse import quote

import dash_mantine_components as dmc
from dash import Input, Output, dcc, html

from .. import state
from ..components import kv_table, section, simple_table


def layout(**_kwargs):
    return dmc.Grid([
        dmc.GridCol(dmc.Paper([
            section("Datasets"),
            dmc.Text("WAVE runs (.map files) found under the data roots.", size="xs", c="dimmed", mb="xs"),
            dmc.ScrollArea(html.Div(id="dsp-list"), h="calc(100vh - 170px)"),
        ], className="wp-controls"), span={"base": 12, "md": 3}),
        dmc.GridCol(html.Div(id="dsp-detail"), span={"base": 12, "md": 9}),
    ], gutter="sm")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def register_callbacks(app) -> None:
    @app.callback(Output("dsp-list", "children"), Input("store-ds", "data"))
    def _list(current):
        entries = state.registry().list_datasets()
        if not entries:
            roots = ", ".join(f"{k} = {v}" for k, v in state.config().data_roots.items())
            return dmc.Alert(f"No .map files found under: {roots}. Start the server with "
                             "`waveplot serve <folder>` to point it at your data.", color="yellow")
        items = []
        last_root = None
        for e in entries:
            if e.root != last_root:
                items.append(dmc.Text(e.root, size="xs", fw=700, c="dimmed", mt="xs"))
                last_root = e.root
            when = _dt.datetime.fromtimestamp(e.mtime).strftime("%Y-%m-%d %H:%M")
            items.append(dmc.NavLink(
                label=e.relative, description=f"{when} · map {_fmt_size(e.size)}",
                href=f"/?ds={quote(e.id, safe='')}", active=e.id == current, variant="light",
                leftSection=html.Span("▤")))
        return items

    @app.callback(Output("dsp-detail", "children"), Input("store-ds", "data"), Input("store-prefs", "data"))
    def _detail(ds_id, prefs):
        ds = state.get_dataset(ds_id)
        if ds is None:
            return dmc.Paper(dmc.Stack([
                dmc.Title("Welcome to WavePlot", order=3),
                dmc.Text("Pick a dataset on the left (or in the header) to explore its snapshots, dumps, "
                         "histories and geometry."),
                dmc.Text("Tip: press ? anywhere for keyboard shortcuts.", size="sm", c="dimmed"),
            ]))
        info = ds.info()
        c = info["counts"]
        geom = ds.geometry
        run_rows = [
            ("Dataset", info["name"]),
            ("Grid", f"{info['grid_cells'][0]} × {info['grid_cells'][1]} × {info['grid_cells'][2]} cells"
             if info.get("grid_cells") else "—"),
            ("Spacing", " / ".join(f"{info['spacing_m'][a]:g}" for a in "ijk") + " m (i / j / k)"),
            ("Model", "3-D" if info.get("model3d") else "2-D" if info.get("model3d") is not None else "—"),
            ("Timestep", f"{info['timestep_s']:g} s" if info.get("timestep_s") else "—"),
            ("Final cycle", f"{info['final_cycle'][0]} (t = {info['final_cycle'][1]:g} s)"
             if info.get("final_cycle") else "—"),
            ("Contents", f"{c['snapshots']} snapshots ({c['series']} series), {c['histories']} histories "
                         f"({c['receivers']} receivers), {c['dumps']} dumps ({c['dump_series']} planes), "
                         f"{c['crack_dumps']} crack dumps"),
        ]
        if geom and geom.physical_size:
            run_rows.append(("Model size", " × ".join(f"{v:g}" for v in geom.physical_size) + " m"))
        if geom and geom.boundary_stress:
            run_rows.append(("Boundary stress", ", ".join(f"{k} {v:.3g}" for k, v in geom.boundary_stress.items()
                                                          if v)))
        issues = [dmc.Alert(str(i), color="red" if i.severity == "error" else "yellow", py=4, mb=4)
                  for i in ds.issues]

        mats = info.get("materials", [])
        mat_table = simple_table(["#", "vp (m/s)", "vs (m/s)", "ρ (kg/m³)", "K (GPa)", "G (GPa)", "ν"],
                                 [[m["number"], f"{m['vp']:.0f}", f"{m['vs']:.0f}", f"{m['rho']:.0f}",
                                   f"{m['K'] / 1e9:.3g}", f"{m['G'] / 1e9:.3g}", f"{m['poisson']:.3f}"]
                                  for m in mats]) if mats else dmc.Text("No material properties", size="sm")

        series_rows = []
        for s in ds.series.values():
            lo, hi = s.global_range()
            href = f"/snapshots?series={quote(s.slug, safe='')}"
            series_rows.append([dcc.Link(s.slug, href=href), s.long_name, f"{s.shape[0]}×{s.shape[1]}", len(s),
                                f"{s.times[0]:.4g}–{s.times[-1]:.4g}", f"{lo:.3g}", f"{hi:.3g}", s.units,
                                f"stope {s.stope_id}" if s.stope_id else ""])
        hist_rows = [[dcc.Link(str(h.number), href=f"/histories?h={h.number}"), h.name, h.units,
                      "global" if h.is_global else f"({h.position[0]},{h.position[1]},{h.position[2]})",
                      h.n, f"{h.dt:.4g}", f"{h.tstart:.4g}–{h.tend:.4g} {h.x_unit}", f"{h.maxval:.3g}",
                      f"{h.minval:.3g}"] for h in ds.histories]
        dump_rows = [[dcc.Link(s.slug, href=f"/dumps?plane={quote(s.slug, safe='')}"), s.label,
                      f"{s.times[0]:.4g}–{s.times[-1]:.4g}"] for s in ds.dump_series.values()]
        files = [[v["name"], _fmt_size(v["size"])] for v in info["files"].values()]
        sc = ds.script
        script_text = "\n".join(sc.commands) if sc else ""

        tabs = dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab(f"Snapshot series ({len(ds.series)})", value="series"),
                dmc.TabsTab(f"Histories ({len(ds.histories)})", value="hist"),
                dmc.TabsTab(f"Dumps ({len(ds.dump_series)})", value="dumps"),
                dmc.TabsTab("Files", value="files"),
                dmc.TabsTab("Input script", value="script", disabled=not script_text),
            ]),
            dmc.TabsPanel(html.Div(simple_table(
                ["Series", "Quantity", "Size", "Frames", "Time (s)", "Min", "Max", "Units", ""], series_rows),
                className="wp-table-scroll"), value="series", pt="xs"),
            dmc.TabsPanel(html.Div(simple_table(
                ["#", "Variable", "Units", "Position (i,j,k)", "Samples", "dt", "Range", "Max", "Min"], hist_rows),
                className="wp-table-scroll"), value="hist", pt="xs"),
            dmc.TabsPanel(simple_table(["Plane", "Description", "Time (s)"], dump_rows)
                          if dump_rows else dmc.Text("No dumps in this run.", size="sm", c="dimmed"),
                          value="dumps", pt="xs"),
            dmc.TabsPanel(simple_table(["File", "Size"], files), value="files", pt="xs"),
            dmc.TabsPanel(dmc.CodeHighlight(code=script_text, language="bash") if script_text else None,
                          value="script", pt="xs"),
        ], value="series")

        quick = dmc.Group([
            dmc.Anchor(dmc.Button("Snapshots ▸", size="xs", variant="light"), href="/snapshots",
                       underline=False) if ds.series else None,
            dmc.Anchor(dmc.Button("Histories ▸", size="xs", variant="light"), href="/histories",
                       underline=False) if ds.histories else None,
            dmc.Anchor(dmc.Button("Dumps ▸", size="xs", variant="light"), href="/dumps",
                       underline=False) if ds.dumps else None,
            dmc.Anchor(dmc.Button("Geometry ▸", size="xs", variant="light"), href="/geometry",
                       underline=False) if ds.geometries else None,
        ], gap="xs")

        return dmc.Stack([
            dmc.Paper(dmc.Grid([
                dmc.GridCol([section("Run"), kv_table(run_rows), html.Div(issues, style={"marginTop": 8}), quick],
                            span={"base": 12, "lg": 7}),
                dmc.GridCol([section("Materials"), mat_table], span={"base": 12, "lg": 5}),
            ])),
            dmc.Paper(tabs),
        ], gap="sm")
