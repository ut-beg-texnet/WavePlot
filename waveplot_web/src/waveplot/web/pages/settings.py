"""Settings page: preferences, legacy WVPL407.SET import, data roots and cache management."""

from __future__ import annotations

import base64
import importlib.metadata as md

import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html, no_update

from ...io.legacy_settings import read_legacy_settings
from .. import state
from ..components import kv_table, section, simple_table
from ..prefs import Preferences, preferences_from_legacy, prefs_path, save_preferences


def layout(**_kwargs):
    return dmc.Grid([
        dmc.GridCol(dmc.Stack([
            dmc.Paper([
                section("Preferences"),
                dmc.Text("Numerics mode and theme are set in the header. These defaults pre-fill the Dumps (ESS "
                         "angles) and Histories (Fourier settings) pages.", size="sm", c="dimmed"),
                html.Div(id="set-prefs-view"),
                dmc.Group([
                    dmc.Button("Save as default for this computer", id="set-save", size="xs", variant="light"),
                    dmc.Button("Reset to defaults", id="set-reset", size="xs", variant="subtle", color="red"),
                ], mt="xs"),
                html.Div(id="set-save-msg"),
            ]),
            dmc.Paper([
                section("Import WavePlot v407 settings (WVPL407.SET)"),
                dcc.Upload(id="set-upload", children=dmc.Paper(dmc.Text("Drop a WVPL407.SET file here or click to "
                                                                         "choose", ta="center", size="sm"),
                                                                p="lg", withBorder=True,
                                                                style={"borderStyle": "dashed", "cursor": "pointer"}),
                           multiple=False),
                html.Div(id="set-legacy-view"),
                dmc.Button("Apply to preferences", id="set-apply", size="xs", mt="xs", disabled=True),
                dcc.Store(id="set-legacy-store"),
            ]),
        ], gap="sm"), span={"base": 12, "md": 7}),
        dmc.GridCol(dmc.Stack([
            dmc.Paper([section("Data roots"), html.Div(id="set-roots"),
                       dmc.Button("Rescan for datasets", id="set-rescan", size="xs", variant="light", mt="xs"),
                       html.Div(id="set-rescan-msg")]),
            dmc.Paper([section("Cache"), html.Div(id="set-cache"),
                       dmc.Button("Clear cache", id="set-clear", size="xs", variant="light", mt="xs")]),
            dmc.Paper([section("About"), html.Div(id="set-about")]),
        ], gap="sm"), span={"base": 12, "md": 5}),
    ], gutter="sm")


def register_callbacks(app) -> None:
    @app.callback(Output("set-prefs-view", "children"), Input("store-prefs", "data"))
    def _view(prefs):
        p = Preferences.model_validate({**Preferences().model_dump(), **(prefs or {})})
        f = p.fourier
        return kv_table([
            ("Numerics", p.mode), ("Theme", p.scheme),
            ("Colour levels", f"{p.n_levels} ({p.level_method})"),
            ("ESS plane / friction", f"θ = {p.ess_theta:g}°, φ = {p.ess_friction:g}°"),
            ("Fourier", f"oversize 2^{f.get('oversize', 1):g}, band {f.get('low_freq', 10):g}–"
                        f"{f.get('high_freq', 100):g} % Nyquist, window {f.get('low_taper', 5):g}–"
                        f"{f.get('high_taper', 95):g} %, taper {f.get('max_taper', 5):g} %"),
            ("Saved defaults file", str(prefs_path())),
        ])

    @app.callback(Output("set-save-msg", "children"), Output("store-prefs", "data", allow_duplicate=True),
                  Input("set-save", "n_clicks"), Input("set-reset", "n_clicks"), State("store-prefs", "data"),
                  prevent_initial_call=True)
    def _save(n_save, n_reset, prefs):
        from dash import callback_context

        if callback_context.triggered_id == "set-reset":
            p = Preferences()
            return dmc.Text("Preferences reset", size="xs", c="dimmed"), p.model_dump()
        p = Preferences.model_validate({**Preferences().model_dump(), **(prefs or {})})
        path = save_preferences(p)
        return dmc.Text(f"Saved to {path}", size="xs", c="dimmed"), no_update

    @app.callback(Output("set-legacy-view", "children"), Output("set-legacy-store", "data"),
                  Output("set-apply", "disabled"), Input("set-upload", "contents"), State("set-upload", "filename"),
                  prevent_initial_call=True)
    def _upload(contents, filename):
        if not contents:
            return no_update, no_update, True
        try:
            raw = base64.b64decode(contents.split(",", 1)[1])
            s = read_legacy_settings(raw)
        except Exception as exc:
            return dmc.Alert(f"Cannot read {filename}: {exc}", color="red"), None, True
        rows = [("File", filename), ("Data", f"{s.drive}:{s.directory}  ({s.fname})"),
                ("Snapshot method", f"{s.snap_set}, levels {s.qty_levels} ({s.snap_fill}), abs {s.snap_col_abs}"),
                ("Fixed min/max", f"{s.vect_set}: {s.min_v:g} … {s.max_v:g}"),
                ("ESS", f"θ = {s.theta:g}°, φ = {s.friction:g}°"), ("Dump type", s.dump_set),
                ("Fourier", f"oversize {s.freq_wind}, band {s.low_freq}–{s.high_freq} %, window {s.low_taper}–"
                            f"{s.high_taper} %, taper {s.max_taper} %"),
                ("Titles", f"{s.hd_title!r} / {s.hz_title!r} / {s.vt_title!r}"),
                ("Multi-panel", f"{s.mult_snap} ({s.whqty}×{s.wvqty}, files {s.mult_file})"),
                ("Colours", f"{s.vidmode}, {s.col_scale}")]
        return kv_table(rows), {"raw": base64.b64encode(raw).decode()}, False

    @app.callback(Output("store-prefs", "data", allow_duplicate=True), Input("set-apply", "n_clicks"),
                  State("set-legacy-store", "data"), State("store-prefs", "data"), prevent_initial_call=True)
    def _apply(_n, legacy, prefs):
        if not legacy:
            return no_update
        s = read_legacy_settings(base64.b64decode(legacy["raw"]))
        base = Preferences.model_validate({**Preferences().model_dump(), **(prefs or {})})
        return preferences_from_legacy(s, base).model_dump()

    @app.callback(Output("set-roots", "children"), Output("set-rescan-msg", "children"),
                  Input("set-rescan", "n_clicks"))
    def _roots(n):
        reg = state.registry()
        entries = reg.list_datasets(refresh=bool(n))
        counts = {}
        for e in entries:
            counts[e.root] = counts.get(e.root, 0) + 1
        rows = [[name, str(path), counts.get(name, 0)] for name, path in reg.roots.items()]
        msg = dmc.Text(f"{len(entries)} dataset(s) found", size="xs", c="dimmed") if n else None
        return simple_table(["Name", "Folder", "Datasets"], rows, fz="sm"), msg

    @app.callback(Output("set-cache", "children"), Input("set-clear", "n_clicks"))
    def _cache(n):
        reg = state.registry()
        if n:
            reg.cache.clear()
        return kv_table([("Cached derived data", f"{reg.cache.currsize / 2**20:.1f} MB"),
                         ("Open datasets", str(len(reg._open)))])

    @app.callback(Output("set-about", "children"), Input("set-about", "id"))
    def _about(_):
        from ... import __version__

        pkgs = ["dash", "dash-mantine-components", "plotly", "numpy", "scipy", "xarray", "matplotlib"]
        rows = [("WavePlot web", __version__)]
        for p in pkgs:
            try:
                rows.append((p, md.version(p)))
            except md.PackageNotFoundError:
                pass
        return kv_table(rows)
