"""Histories page: receiver time series with per-trace processing pipelines (modern expressions
or legacy v407 formulas), stacked/overlaid display, spectra, spectrograms, response spectra,
record sections, metrics and exports."""

from __future__ import annotations

import uuid
from urllib.parse import parse_qs

import dash_mantine_components as dmc
import numpy as np
from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update

from ...formula.evaluate import EvalResult, evaluate_text, parse_any
from ...formula.expr import parse_expr
from ...formula.nodes import Call
from ...formula.ops import OPS, FormulaError, ops_by_category
from ...formula.printer import fmt_value, to_expr
from ...render.history_fig import (
    SCALES,
    TraceView,
    record_section_figure,
    response_figure,
    spectra_figure,
    spectrogram_figure,
    traces_figure,
)
from ...render.theme import PALETTE, empty_figure, si_format
from ...science.signal import FourierSettings
from .. import state
from ..components import labeled, provenance_badge, section, simple_table

SCALE_DATA = [{"value": k, "label": v} for k, v in SCALES.items()]


def layout(h: str | None = None, **_kwargs):
    controls = dmc.Stack([
        section("Add histories"),
        dmc.MultiSelect(id="hst-pick", data=[], searchable=True, placeholder="Pick histories (grouped by receiver)",
                        comboboxProps={"withinPortal": True}, maxDropdownHeight=380, hidePickedOptions=True),
        dmc.Group([dmc.Button("Add", id="hst-add", size="xs"),
                   dmc.Button("Clear all", id="hst-clear", size="xs", variant="subtle", color="red")], gap="xs"),
        section("Formula"),
        dmc.Textarea(id="hst-formula", autosize=True, minRows=1, maxRows=4,
                     placeholder="h2 | integrate | bandpass(1, 10)   or legacy  [2]I&f,1,10",
                     className="wp-formula"),
        dmc.Select(id="hst-self", data=[], placeholder="slot history ('self', optional)", clearable=True,
                   searchable=True, comboboxProps={"withinPortal": True}),
        html.Div(id="hst-formula-preview"),
        dmc.Group([dmc.Button("Add formula trace", id="hst-add-formula", size="xs", variant="light"),
                   dmc.Button("Load form.N files", id="hst-load-forms", size="xs", variant="subtle")], gap="xs"),
        section("Display"),
        dmc.SegmentedControl(id="hst-layout", data=[{"label": "Stacked", "value": "stacked"},
                                                    {"label": "Overlay", "value": "overlay"}], value="stacked",
                             fullWidth=True),
        labeled("Vertical scale", dmc.SegmentedControl(
            id="hst-vlink", data=[{"label": "Own", "value": "own"}, {"label": "Linked", "value": "linked"},
                                  {"label": "Fixed", "value": "fixed"}], value="own", fullWidth=True),
            "Own: each trace autoscaled. Linked: common ±max. Fixed: ±your value (v407 'Set' scales)."),
        dmc.NumberInput(id="hst-vfix", placeholder="fixed ± value", decimalScale=8, disabled=True),
        labeled("Time axis", dmc.SegmentedControl(
            id="hst-hlink", data=[{"label": "Common", "value": "common"}, {"label": "Own", "value": "own"},
                                  {"label": "Window", "value": "window"}], value="common", fullWidth=True)),
        dmc.Group([dmc.NumberInput(id="hst-t1", placeholder="t1 (s)", w=110, disabled=True, decimalScale=6),
                   dmc.NumberInput(id="hst-t2", placeholder="t2 (s)", w=110, disabled=True, decimalScale=6)],
                  gap="xs", grow=True),
        dmc.Switch(id="hst-markers", label="Show sample markers", checked=False),
        dmc.Accordion(dmc.AccordionItem([
            dmc.AccordionControl("Fourier settings (spectrum operation defaults)"),
            dmc.AccordionPanel(dmc.Stack([
                dmc.NumberInput(id="hst-f-over", label="FFT oversize (2^n)", value=1, min=0, max=4),
                dmc.Group([dmc.NumberInput(id="hst-f-lo", label="Band start %Nyq", value=10, w=120),
                           dmc.NumberInput(id="hst-f-hi", label="Band end %Nyq", value=100, w=120)], gap="xs"),
                dmc.Group([dmc.NumberInput(id="hst-t-lo", label="Window start %", value=5, w=80),
                           dmc.NumberInput(id="hst-t-hi", label="Window end %", value=95, w=80),
                           dmc.NumberInput(id="hst-t-max", label="Taper %", value=5, w=80)], gap="xs"),
            ], gap=4)),
        ], value="fourier"), variant="contained", chevronPosition="right"),
        section("Export"),
        dmc.Group([
            dmc.Button("CSV", id="hst-dl-csv", size="xs", variant="default"),
            dmc.Button("ASC", id="hst-dl-asc", size="xs", variant="default"),
            dmc.Button("ASC (no time)", id="hst-dl-asc0", size="xs", variant="default"),
            dmc.Button("TBL", id="hst-dl-tbl", size="xs", variant="default"),
        ], gap=6),
        dcc.Download(id="hst-download"),
    ], gap=6)

    tabs = dmc.Tabs([
        dmc.TabsList([
            dmc.TabsTab("Traces", value="traces"), dmc.TabsTab("Spectra", value="spectra"),
            dmc.TabsTab("Spectrogram", value="spectrogram"), dmc.TabsTab("Response spectra", value="response"),
            dmc.TabsTab("Record section", value="section"), dmc.TabsTab("Metrics", value="metrics"),
        ]),
        dmc.TabsPanel([
            dmc.Group([html.Div(id="hst-badge"), html.Div(id="hst-msg")], justify="space-between", mt=4),
            dcc.Loading(dcc.Graph(id="hst-graph", className="wp-graph-short",
                                  config={"displaylogo": False, "scrollZoom": True,
                                          "toImageButtonOptions": {"scale": 2, "filename": "waveplot_histories"}}),
                        delay_show=400, type="dot"),
            html.Div(id="hst-list"),
        ], value="traces"),
        dmc.TabsPanel([
            dmc.Group([dmc.SegmentedControl(id="hst-spec-kind",
                                            data=[{"label": "Fourier amplitude", "value": "fourier"},
                                                  {"label": "Power (Welch)", "value": "psd"}], value="fourier"),
                       dmc.NumberInput(id="hst-spec-smooth", label=None, value=1, min=1, max=51, w=120,
                                       leftSection="smooth", leftSectionWidth=60)], mt=4),
            dcc.Graph(id="hst-spec-graph", className="wp-graph-short", config={"displaylogo": False}),
        ], value="spectra"),
        dmc.TabsPanel([
            dmc.Select(id="hst-sg-trace", data=[], placeholder="trace", w=420, mt=4,
                       comboboxProps={"withinPortal": True}),
            dcc.Graph(id="hst-sg-graph", className="wp-graph-short", config={"displaylogo": False}),
        ], value="spectrogram"),
        dmc.TabsPanel([
            dmc.Group([
                dmc.SegmentedControl(id="hst-rs-input", data=[{"label": "auto", "value": "auto"},
                                                              {"label": "acceleration", "value": "acc"},
                                                              {"label": "velocity", "value": "vel"},
                                                              {"label": "displacement", "value": "disp"}],
                                     value="auto"),
                dmc.NumberInput(id="hst-rs-damp", value=5, min=0, max=50, w=110, leftSection="ζ %",
                                leftSectionWidth=40),
                dmc.SegmentedControl(id="hst-rs-q", data=[{"label": "PSA", "value": "psa"},
                                                          {"label": "PSV", "value": "psv"},
                                                          {"label": "SD", "value": "sd"}], value="psa"),
            ], mt=4, gap="xs"),
            dcc.Graph(id="hst-rs-graph", className="wp-graph-short", config={"displaylogo": False}),
        ], value="response"),
        dmc.TabsPanel([
            dmc.Group([
                dmc.Select(id="hst-rec-var", data=[], placeholder="variable", w=160,
                           comboboxProps={"withinPortal": True}),
                dmc.SegmentedControl(id="hst-rec-norm", data=[{"label": "Normalise each", "value": "trace"},
                                                              {"label": "Common scale", "value": "global"}],
                                     value="trace"),
                dmc.NumberInput(id="hst-rec-scale", value=1.0, step=0.25, w=100, leftSection="×", decimalScale=2),
                dmc.NumberInput(id="hst-rec-vred", value=0, w=150, leftSection="v_red", leftSectionWidth=45,
                                rightSection="m/s", rightSectionWidth=40),
                dmc.Switch(id="hst-rec-arr", label="P/S arrivals", checked=True),
            ], mt=4, gap="xs"),
            dcc.Graph(id="hst-rec-graph", className="wp-graph-short", config={"displaylogo": False}),
        ], value="section"),
        dmc.TabsPanel(html.Div(id="hst-metrics", style={"marginTop": 8}), value="metrics"),
    ], value="traces", id="hst-tabs")

    modal = dmc.Modal(id="hst-modal", title="Edit trace", size="xl", children=dmc.Stack([
        dmc.TextInput(id="hst-ed-expr", label="Expression", className="wp-formula"),
        html.Div(id="hst-ed-preview"),
        section("Processing steps"),
        html.Div(id="hst-ed-steps"),
        dmc.Group([
            dmc.Select(id="hst-ed-op", data=_op_options(), placeholder="add a processing step…", searchable=True,
                       w=320, comboboxProps={"withinPortal": True}),
            dmc.Button("Add step", id="hst-ed-addstep", size="xs", variant="light"),
        ], gap="xs", align="flex-end"),
        html.Div(id="hst-ed-params"),
        dmc.Group([dmc.Button("Apply", id="hst-ed-apply"), dmc.Button("Cancel", id="hst-ed-cancel", variant="default")],
                  justify="flex-end"),
        dcc.Store(id="hst-ed-id"),
    ], gap=6))

    return dmc.Grid([
        dmc.GridCol(dmc.Paper(controls, className="wp-controls"), span={"base": 12, "md": 3}),
        dmc.GridCol([dmc.Paper(tabs), modal,
                     dcc.Store(id="hst-traces", storage_type="session"), dcc.Store(id="hst-url-h", data=h)],
                    span={"base": 12, "md": 9}),
    ], gutter="sm")


def _op_options() -> list[dict]:
    return [{"group": cat.capitalize(), "items": [{"value": s.name, "label": f"{s.title} — {s.name}"}
                                                   for s in specs]}
            for cat, specs in ops_by_category(1).items()]


def _settings(over, flo, fhi, tlo, thi, tmax) -> FourierSettings:
    def num(v, d):
        return float(v) if isinstance(v, int | float) else d
    return FourierSettings(int(num(over, 1)), num(flo, 10.0), num(fhi, 100.0), num(tlo, 5.0), num(thi, 95.0),
                           num(tmax, 5.0))


def _eval(ds, ds_id: str, spec: dict, mode: str, fs: FourierSettings) -> EvalResult:
    reg = state.registry()
    key = ("hist", reg.key_for(ds), spec.get("expr"), spec.get("self"), mode, fs)
    return reg.cache.get_or_compute(key, lambda: evaluate_text(ds, spec.get("expr", ""), spec.get("self"), mode, fs,
                                                               prefer=spec.get("syntax", "auto")))


def _new_trace(expr: str, self_ref: int | None = None, legacy: str | None = None, scale: str = "lin") -> dict:
    return {"id": uuid.uuid4().hex[:8], "expr": expr, "self": self_ref, "legacy": legacy, "scale": scale,
            "visible": True, "syntax": "modern"}


def _pick_options(ds) -> tuple[list, list]:
    groups = []
    for r in ds.receivers:
        groups.append({"group": f"R{r.number} ({r.position[0]},{r.position[1]},{r.position[2]})",
                       "items": [{"value": str(h.number), "label": f"h{h.number} {h.name}"} for h in r.records]})
    other = [h for h in ds.histories if h.is_global or h.is_profile]
    if other:
        groups.insert(0, {"group": "Global / profiles", "items": [{"value": str(h.number),
                                                                   "label": f"h{h.number} {h.name}"
                                                                   + (" (profile)" if h.is_profile else "")}
                                                                  for h in other]})
    flat = [{"value": str(h.number), "label": f"h{h.number} {h.name} ({h.position[0]},{h.position[1]},"
                                               f"{h.position[2]})"} for h in ds.histories]
    return groups, flat


def _steps(expr: str) -> tuple[object, list[Call]] | None:
    """Split ``source | op1 | op2`` into (source node, [ops]) when the expression is such a chain."""
    try:
        node = parse_expr(expr)
    except FormulaError:
        return None
    ops: list[Call] = []
    while isinstance(node, Call) and OPS.get(node.op) and OPS[node.op].arity == 1 and len(node.args) == 1:
        ops.append(node)
        node = node.args[0]
    return node, ops[::-1]


def register_callbacks(app) -> None:
    @app.callback(Output("hst-pick", "data"), Output("hst-self", "data"), Output("hst-rec-var", "data"),
                  Output("hst-rec-var", "value"), Input("store-ds", "data"))
    def _options(ds_id):
        ds = state.get_dataset(ds_id)
        if ds is None:
            return [], [], [], None
        groups, flat = _pick_options(ds)
        names = sorted({h.name for r in ds.receivers for h in r.records})
        return groups, flat, names, (names[0] if names else None)

    # --- trace store reducer -------------------------------------------------------------------
    @app.callback(
        Output("hst-traces", "data"), Output("hst-pick", "value"), Output("hst-formula", "value"),
        Output("hst-msg", "children"),
        Input("hst-add", "n_clicks"), Input("hst-add-formula", "n_clicks"), Input("hst-load-forms", "n_clicks"),
        Input("hst-clear", "n_clicks"), Input({"type": "hst-scale", "index": ALL}, "value"),
        Input({"type": "hst-vis", "index": ALL}, "checked"), Input({"type": "hst-del", "index": ALL}, "n_clicks"),
        Input({"type": "hst-up", "index": ALL}, "n_clicks"), Input("hst-ed-apply", "n_clicks"),
        Input("store-ds", "data"), Input("store-nav", "data"), Input("hst-url-h", "data"),
        State("hst-traces", "data"), State("hst-pick", "value"), State("hst-formula", "value"),
        State("hst-self", "value"), State("hst-ed-id", "data"), State("hst-ed-expr", "value"),
    )
    def _reduce(_a, _af, _lf, _clr, scales, vis, _dels, _ups, _apply, ds_id, nav, url_h, store, picks, formula,
                self_ref, ed_id, ed_expr):
        trig = callback_context.triggered_id
        store = dict(store or {})
        traces = list(store.get("traces", [])) if store.get("ds") == ds_id else []
        changed = store.get("ds") != ds_id
        nav_ts = store.get("nav_ts")
        out_pick, out_formula, msg = no_update, no_update, no_update
        ds = state.get_dataset(ds_id)
        if ds is None:
            return {"ds": ds_id, "traces": [], "nav_ts": nav_ts}, no_update, no_update, None
        if trig is None or trig in ("store-nav", "store-ds", "hst-url-h"):
            # navigation requests: receiver clicks on other pages, ?h= links
            if nav and nav.get("ds") == ds_id and nav.get("open_histories") and nav.get("ts") != nav_ts:
                existing = {t["expr"] for t in traces}
                for h in nav["open_histories"]:
                    if f"h{h}" not in existing:
                        traces.append(_new_trace(f"h{h}", h))
                nav_ts = nav.get("ts")
                changed = True
            if url_h and not traces:
                for h in str(url_h).split(","):
                    if h.strip().isdigit():
                        traces.append(_new_trace(f"h{h.strip()}", int(h)))
                changed = True
        if trig == "hst-add" and picks:
            for p in picks:
                traces.append(_new_trace(f"h{int(p)}", int(p)))
            out_pick, changed = [], True
        elif trig == "hst-add-formula" and formula and formula.strip():
            try:
                node, syntax, _diags = parse_any(formula)
            except FormulaError as exc:
                return store, no_update, no_update, dmc.Alert(str(exc), color="red", py=4)
            sref = int(self_ref) if self_ref else None
            t = _new_trace(to_expr(node), sref, formula.strip() if syntax == "legacy" else None)
            traces.append(t)
            out_formula, changed = "", True
        elif trig == "hst-load-forms":
            found = 0
            for n in range(1, 9):
                p = next((f for f in ds.files.folder.iterdir() if f.name.lower() == f"form.{n}"), None)
                if p is None:
                    continue
                text = p.read_text(errors="replace").splitlines()[:1]
                if text and text[0].strip():
                    node, _syntax, _d = parse_any(text[0])
                    traces.append(_new_trace(to_expr(node), None, text[0].strip()))
                    found += 1
            msg = dmc.Text(f"{found} form.N file(s) loaded" if found else "No form.1 … form.8 files in the dataset "
                           "folder", size="xs", c="dimmed")
            changed = changed or found > 0
        elif trig == "hst-clear":
            traces, changed = [], True
        elif isinstance(trig, dict):
            tid = trig.get("index")
            idx = next((k for k, t in enumerate(traces) if t["id"] == tid), None)
            if idx is not None:
                if trig["type"] == "hst-del":
                    traces.pop(idx)
                    changed = True
                elif trig["type"] == "hst-up" and idx > 0:
                    traces[idx - 1], traces[idx] = traces[idx], traces[idx - 1]
                    changed = True
                elif trig["type"] == "hst-scale":
                    val = next((v for v, s in zip(scales, callback_context.inputs_list[4], strict=False)
                                if s["id"]["index"] == tid), None)
                    if val and val != traces[idx]["scale"]:
                        traces[idx] = {**traces[idx], "scale": val}
                        changed = True
                elif trig["type"] == "hst-vis":
                    val = next((v for v, s in zip(vis, callback_context.inputs_list[5], strict=False)
                                if s["id"]["index"] == tid), None)
                    if val is not None and val != traces[idx]["visible"]:
                        traces[idx] = {**traces[idx], "visible": bool(val)}
                        changed = True
        elif trig == "hst-ed-apply" and ed_id:
            idx = next((k for k, t in enumerate(traces) if t["id"] == ed_id), None)
            if idx is not None and ed_expr:
                try:
                    node, syntax, _d = parse_any(ed_expr)
                    traces[idx] = {**traces[idx], "expr": to_expr(node),
                                   "legacy": ed_expr if syntax == "legacy" else traces[idx].get("legacy")}
                    changed = True
                except FormulaError as exc:
                    msg = dmc.Alert(str(exc), color="red", py=4)
        if not changed:
            return no_update, out_pick, out_formula, msg
        return {"ds": ds_id, "traces": traces, "nav_ts": nav_ts}, out_pick, out_formula, msg

    @app.callback(Output("hst-url-h", "data"), Input("url", "search"))
    def _url_h(search):
        q = parse_qs((search or "").lstrip("?"))
        return (q.get("h") or [None])[0]

    @app.callback(Output("hst-formula-preview", "children"), Input("hst-formula", "value"),
                  State("store-ds", "data"), State("hst-self", "value"), State("store-prefs", "data"))
    def _preview(text, ds_id, self_ref, prefs):
        if not text or not text.strip():
            return dmc.Text("Modern syntax or legacy v407 formulas are accepted. See Help → Formulas.", size="xs",
                            c="dimmed")
        return _preview_block(text, ds_id, self_ref, prefs)

    # --- rendering ---------------------------------------------------------------------------------
    @app.callback(
        Output("hst-graph", "figure"), Output("hst-list", "children"), Output("hst-badge", "children"),
        Output("hst-spec-graph", "figure"), Output("hst-sg-trace", "data"), Output("hst-metrics", "children"),
        Input("hst-traces", "data"), Input("hst-layout", "value"), Input("hst-vlink", "value"),
        Input("hst-vfix", "value"), Input("hst-hlink", "value"), Input("hst-t1", "value"), Input("hst-t2", "value"),
        Input("hst-markers", "checked"), Input("store-prefs", "data"), Input("hst-spec-kind", "value"),
        Input("hst-spec-smooth", "value"),
        Input("hst-f-over", "value"), Input("hst-f-lo", "value"), Input("hst-f-hi", "value"),
        Input("hst-t-lo", "value"), Input("hst-t-hi", "value"), Input("hst-t-max", "value"),
        State("store-ds", "data"),
    )
    def _render(store, lay, vlink, vfix, hlink, t1, t2, markers, prefs, spec_kind, spec_smooth, *rest):
        *fargs, ds_id = rest
        fs = _settings(*fargs)
        scheme, mode = state.scheme_of(prefs), state.mode_of(prefs)
        ds = state.get_dataset(ds_id)
        traces = (store or {}).get("traces", []) if (store or {}).get("ds") == ds_id else []
        if ds is None:
            e = empty_figure("Select a dataset", scheme)
            return e, None, None, e, [], None
        views, cards, quirks = [], [], set()
        for k, spec in enumerate(traces):
            res = _eval(ds, ds_id, spec, mode, fs)
            quirks |= set(res.provenance.quirks)
            col = PALETTE[k % len(PALETTE)]
            if res.trace is not None and spec.get("visible", True):
                views.append(TraceView(res.trace, spec["expr"] if len(spec["expr"]) < 70 else res.trace.label,
                                       spec.get("scale", "lin"), col,
                                       "spectrum" if res.trace.x_unit == "Hz" else ""))
            cards.append(_card(spec, res, col, k))
        win = (float(t1), float(t2)) if isinstance(t1, int | float) and isinstance(t2, int | float) else None
        fig = traces_figure(views, layout=lay or "stacked", vlink=vlink or "own",
                            vfix=float(vfix) if isinstance(vfix, int | float) else None, hlink=hlink or "common",
                            window=win, scheme=scheme, uirevision=f"{ds_id}", markers=bool(markers))
        if len(views) > 4 and (lay or "stacked") == "stacked":
            fig.update_layout(height=max(420, 150 * len(views)))
        spec_fig = spectra_figure(views, spec_kind or "fourier", scheme, int(spec_smooth or 1)) if views else \
            empty_figure("No traces", scheme)
        sg_opts = [{"value": spec["id"], "label": spec["expr"]} for spec in traces]
        metrics = _metrics_table(views)
        return fig, (dmc.Stack(cards, gap=4, mt="xs") if cards else None), provenance_badge(sorted(quirks), mode), \
            spec_fig, sg_opts, metrics

    @app.callback(Output("hst-f-over", "value"), Output("hst-f-lo", "value"), Output("hst-f-hi", "value"),
                  Output("hst-t-lo", "value"), Output("hst-t-hi", "value"), Output("hst-t-max", "value"),
                  Input("store-prefs", "data"))
    def _fourier_defaults(prefs):
        f = (prefs or {}).get("fourier") or {}
        return (f.get("oversize", 1), f.get("low_freq", 10), f.get("high_freq", 100), f.get("low_taper", 5),
                f.get("high_taper", 95), f.get("max_taper", 5))

    @app.callback(Output("hst-vfix", "disabled"), Output("hst-t1", "disabled"), Output("hst-t2", "disabled"),
                  Input("hst-vlink", "value"), Input("hst-hlink", "value"))
    def _enable(vlink, hlink):
        return vlink != "fixed", hlink != "window", hlink != "window"

    @app.callback(Output("hst-sg-graph", "figure"), Input("hst-sg-trace", "value"), Input("hst-traces", "data"),
                  Input("store-prefs", "data"), State("store-ds", "data"))
    def _spectrogram(tid, store, prefs, ds_id):
        scheme, mode = state.scheme_of(prefs), state.mode_of(prefs)
        ds = state.get_dataset(ds_id)
        traces = (store or {}).get("traces", [])
        spec = next((t for t in traces if t["id"] == tid), traces[0] if traces else None)
        if ds is None or spec is None:
            return empty_figure("Add traces first", scheme)
        res = _eval(ds, ds_id, spec, mode, FourierSettings())
        if res.trace is None:
            return empty_figure(res.error or "cannot evaluate", scheme)
        return spectrogram_figure(TraceView(res.trace, spec["expr"]), scheme)

    @app.callback(Output("hst-rs-graph", "figure"), Input("hst-traces", "data"), Input("hst-rs-input", "value"),
                  Input("hst-rs-damp", "value"), Input("hst-rs-q", "value"), Input("store-prefs", "data"),
                  State("store-ds", "data"))
    def _response(store, kind, damp, q, prefs, ds_id):
        from ...science.metrics import response_spectrum
        from ...science.signal import differentiate

        scheme, mode = state.scheme_of(prefs), state.mode_of(prefs)
        ds = state.get_dataset(ds_id)
        traces = (store or {}).get("traces", [])
        if ds is None or not traces:
            return empty_figure("Add traces first", scheme)
        results = []
        units = ""
        for spec in traces:
            res = _eval(ds, ds_id, spec, mode, FourierSettings())
            t = res.trace
            if t is None or t.x_unit != "s" or t.n < 16:
                continue
            k = kind
            if k == "auto":
                k = {"m/s": "vel", "m": "disp"}.get(t.y_unit, "acc")
            acc = t.y
            if k in ("vel", "disp"):
                acc = differentiate(acc, t.dt)
            if k == "disp":
                acc = differentiate(acc, t.dt)
            tmin = max(2.5 * t.dt, 1e-4)
            tmax = max(t.t_end - t.t0, 10 * tmin) / 2
            periods = np.geomspace(tmin, tmax, 60)
            results.append((spec["expr"], response_spectrum(acc, t.dt, periods, float(damp or 5) / 100)))
            units = {"psa": "(m/s²)" if t.y_unit in ("m/s", "m", "m/s²") else "",
                     "psv": "(m/s)" if t.y_unit in ("m/s", "m", "m/s²") else "",
                     "sd": "(m)" if t.y_unit in ("m/s", "m", "m/s²") else ""}[q or "psa"]
        if not results:
            return empty_figure("Response spectra need time-domain traces", scheme)
        return response_figure(results, q or "psa", scheme, units)

    @app.callback(Output("hst-rec-graph", "figure"), Input("hst-rec-var", "value"), Input("hst-rec-norm", "value"),
                  Input("hst-rec-scale", "value"), Input("hst-rec-vred", "value"), Input("hst-rec-arr", "checked"),
                  Input("store-prefs", "data"), State("store-ds", "data"))
    def _section(var, norm, scale, vred, arr, prefs, ds_id):
        from ...formula.evaluate import make_loader

        scheme = state.scheme_of(prefs)
        ds = state.get_dataset(ds_id)
        if ds is None or not var:
            return empty_figure("Select a variable", scheme)
        geom = ds.geometry
        sp = ds.spacing
        if geom is not None and geom.sources:
            ci, cj, ck = geom.source_centroid()
        else:
            ci = cj = ck = 0
        load = make_loader(ds, "standard")
        rows = []
        for r in ds.receivers:
            h = r.by_name(var)
            if h is None:
                continue
            i, j, k = r.position
            d = float(np.sqrt(((i - ci) * sp["i"]) ** 2 + ((j - cj) * sp["j"]) ** 2 + ((k - ck) * sp["k"]) ** 2))
            rows.append((d, load(h.number), f"R{r.number} h{h.number}"))
        arrivals = None
        if arr and geom is not None and geom.props:
            p = geom.props[0]
            arrivals = {"P": p.vp, "S": p.vs}
        return record_section_figure(rows, scale=float(scale or 1), norm=norm or "trace",
                                     v_reduce=float(vred or 0), arrivals=arrivals, scheme=scheme)

    # --- trace editor (modal) -----------------------------------------------------------------------------
    @app.callback(
        Output("hst-modal", "opened"), Output("hst-ed-id", "data"), Output("hst-ed-expr", "value"),
        Input({"type": "hst-edit", "index": ALL}, "n_clicks"), Input("hst-ed-cancel", "n_clicks"),
        Input("hst-ed-apply", "n_clicks"), Input("hst-ed-addstep", "n_clicks"),
        Input({"type": "hst-ed-rm", "index": ALL}, "n_clicks"),
        State("hst-traces", "data"), State("hst-ed-expr", "value"), State("hst-ed-op", "value"),
        State({"type": "hst-ed-param", "name": ALL}, "value"), State({"type": "hst-ed-param", "name": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _modal(edit_clicks, _c, _a, _add, rm_clicks, store, expr, op_name, pvals, pids):
        trig = callback_context.triggered_id
        if isinstance(trig, dict) and trig.get("type") == "hst-edit":
            if not any(edit_clicks):
                return no_update, no_update, no_update
            spec = next((t for t in (store or {}).get("traces", []) if t["id"] == trig["index"]), None)
            if spec is None:
                return no_update, no_update, no_update
            return True, spec["id"], spec["expr"]
        if trig in ("hst-ed-cancel", "hst-ed-apply"):
            return False, no_update, no_update
        if trig == "hst-ed-addstep" and op_name in OPS and expr:
            params = {}
            for v, pid in zip(pvals, pids, strict=False):
                if v not in (None, ""):
                    params[pid["name"]] = v
            spec = OPS[op_name]
            args = ", ".join(f"{k}={fmt_value(_coerce(spec, k, v))}" for k, v in params.items())
            base = expr.strip()
            try:
                node = parse_expr(base)
                base = to_expr(node)
                if isinstance(node, Call) and OPS.get(node.op) and OPS[node.op].infix:
                    base = f"({base})"
            except FormulaError:
                pass
            return no_update, no_update, f"{base} | {op_name}" + (f"({args})" if args else "")
        if isinstance(trig, dict) and trig.get("type") == "hst-ed-rm" and any(rm_clicks or []):
            parsed = _steps(expr or "")
            if parsed is None:
                return no_update, no_update, no_update
            src, ops = parsed
            k = int(trig["index"])
            if 0 <= k < len(ops):
                node = src
                for n, o in enumerate(ops):
                    if n != k:
                        node = Call(o.op, (node,), o.params)
                return no_update, no_update, to_expr(node)
        return no_update, no_update, no_update

    @app.callback(Output("hst-ed-steps", "children"), Output("hst-ed-preview", "children"),
                  Input("hst-ed-expr", "value"), State("store-ds", "data"), State("hst-ed-id", "data"),
                  State("hst-traces", "data"), State("store-prefs", "data"))
    def _editor_view(expr, ds_id, ed_id, store, prefs):
        if not expr:
            return None, None
        spec = next((t for t in (store or {}).get("traces", []) if t["id"] == ed_id), {})
        parsed = _steps(expr)
        if parsed is None:
            steps = dmc.Text("Not a simple chain — edit the expression directly (legacy formulas are converted on "
                             "Apply).", size="xs", c="dimmed")
        else:
            src, ops = parsed
            chips = [dmc.Badge(to_expr(src), variant="outline", size="lg", tt="none")]
            for k, o in enumerate(ops):
                label = o.op + (f"({', '.join(f'{a}={fmt_value(b)}' for a, b in o.params)})" if o.params else "")
                chips.append(html.Span("→", style={"margin": "0 4px"}))
                chips.append(dmc.Badge(label, size="lg", variant="light", tt="none", rightSection=dmc.ActionIcon(
                    "×", id={"type": "hst-ed-rm", "index": k}, size="xs", variant="transparent")))
            steps = dmc.Group(chips, gap=2)
        return steps, _preview_block(expr, ds_id, spec.get("self"), prefs)

    @app.callback(Output("hst-ed-params", "children"), Input("hst-ed-op", "value"))
    def _param_form(op_name):
        spec = OPS.get(op_name or "")
        if spec is None:
            return None
        fields = []
        for p in spec.params:
            pid = {"type": "hst-ed-param", "name": p.name}
            label = f"{p.name}" + (f" ({p.unit})" if p.unit else "")
            if p.kind == "choice":
                fields.append(dmc.Select(id=pid, label=label, data=[str(c) for c in p.choices],
                                         value=str(p.default) if p.default is not None else None, w=140,
                                         description=p.doc, comboboxProps={"withinPortal": True}, clearable=True))
            elif p.kind == "str":
                fields.append(dmc.TextInput(id=pid, label=label, value=p.default or "", w=140))
            else:
                fields.append(dmc.NumberInput(id=pid, label=label, value=p.default, w=140, description=p.doc,
                                              decimalScale=8 if p.kind == "float" else 0))
        doc = dmc.Text(spec.doc or spec.title, size="xs", c="dimmed")
        legacy = dmc.Text(f"v407: {spec.legacy}", size="xs", c="dimmed") if spec.legacy else None
        return dmc.Stack([doc, legacy, dmc.Group(fields, gap="xs", align="flex-start")], gap=4)

    # --- exports -------------------------------------------------------------------------------------------
    @app.callback(Output("hst-download", "data"),
                  Input("hst-dl-csv", "n_clicks"), Input("hst-dl-asc", "n_clicks"), Input("hst-dl-asc0", "n_clicks"),
                  Input("hst-dl-tbl", "n_clicks"), State("hst-traces", "data"), State("store-ds", "data"),
                  State("store-prefs", "data"), prevent_initial_call=True)
    def _export(_c, _a, _a0, _t, store, ds_id, prefs):
        from ...export.tables import legacy_asc, legacy_tbl, traces_csv

        ds = state.get_dataset(ds_id)
        mode = state.mode_of(prefs)
        traces = [r.trace for spec in (store or {}).get("traces", [])
                  if (r := _eval(ds, ds_id, spec, mode, FourierSettings())).trace is not None] if ds else []
        if not traces:
            return no_update
        trig = callback_context.triggered_id
        base = f"{ds.name}_histories"
        if trig == "hst-dl-csv":
            return dcc.send_string(traces_csv(traces), f"{base}.csv")
        if trig == "hst-dl-tbl":
            return dcc.send_string(legacy_tbl(traces[0]), f"{ds.name}.tbl")
        return dcc.send_string(legacy_asc(traces, with_time=trig == "hst-dl-asc", mode=mode), f"{ds.name}.asc")


def _coerce(spec, name, v):
    p = next((p for p in spec.params if p.name == name), None)
    if p is None:
        return v
    if p.kind == "int":
        return int(float(v))
    if p.kind == "float":
        return float(v)
    return v


def _preview_block(text: str, ds_id, self_ref, prefs):
    ds = state.get_dataset(ds_id)
    mode = state.mode_of(prefs)
    try:
        node, syntax, diags = parse_any(text)
    except FormulaError as exc:
        return dmc.Alert(str(exc), color="red", py=4)
    modern = to_expr(node)
    items = [dmc.Text([dmc.Badge(syntax, size="xs", color="orange" if syntax == "legacy" else "teal",
                                 variant="light", mr=6, tt="none"), html.Code(modern, className="wp-mono")], size="xs")]
    for d in diags:
        items.append(dmc.Text(str(d), size="xs", c="red" if d.severity == "error" else "orange"))
    if ds is not None:
        res = evaluate_text(ds, text, int(self_ref) if self_ref else None, mode)
        if res.error:
            items.append(dmc.Text(f"cannot evaluate: {res.error}", size="xs", c="red"))
        elif res.trace is not None:
            t = res.trace
            items.append(dmc.Text(f"→ {t.n} samples, {t.x_unit} axis {t.t0:.4g} … {t.t_end:.4g}, peak "
                                  f"{si_format(float(np.nanmax(np.abs(t.y))), t.y_unit)}", size="xs", c="dimmed"))
            for w in res.diagnostics:
                if w not in [str(d) for d in diags]:
                    items.append(dmc.Text(w, size="xs", c="orange"))
    return dmc.Stack(items, gap=2)


def _card(spec: dict, res: EvalResult, color: str, k: int):
    tid = spec["id"]
    t = res.trace
    info = (f"{t.n} samples · {t.x_unit} · peak {si_format(float(np.nanmax(np.abs(t.y))), t.y_unit)}"
            if t is not None and t.n and np.isfinite(t.y).any() else "")
    status = dmc.Text(res.error, size="xs", c="red") if res.error else dmc.Text(info, size="xs", c="dimmed")
    extra = []
    if spec.get("legacy"):
        extra.append(dmc.Tooltip(dmc.Badge("v407 formula", size="xs", variant="outline", color="orange", tt="none"),
                                 label=f"original: {spec['legacy']}"))
    for w in (res.diagnostics or [])[:3]:
        extra.append(dmc.Text(w, size="xs", c="orange"))
    return dmc.Paper(dmc.Group([
        dmc.Group([
            html.Div(style={"width": 10, "height": 28, "background": color, "borderRadius": 3}),
            dmc.Stack([dmc.Group([dmc.Text(f"[{k + 1}]", size="xs", fw=700),
                                  html.Code(spec["expr"], className="wp-mono"), *extra], gap=6),
                       status], gap=0),
        ], gap="xs", wrap="nowrap", style={"minWidth": 0, "flex": 1}),
        dmc.Group([
            dmc.Select(id={"type": "hst-scale", "index": tid}, data=SCALE_DATA, value=spec.get("scale", "lin"), w=105,
                       size="xs", comboboxProps={"withinPortal": True}, allowDeselect=False),
            dmc.Switch(id={"type": "hst-vis", "index": tid}, checked=spec.get("visible", True), size="xs"),
            dmc.Tooltip(dmc.ActionIcon("✎", id={"type": "hst-edit", "index": tid}, variant="default", size="md"),
                        label="Edit processing"),
            dmc.Tooltip(dmc.ActionIcon("↑", id={"type": "hst-up", "index": tid}, variant="default", size="md"),
                        label="Move up"),
            dmc.Tooltip(dmc.ActionIcon("×", id={"type": "hst-del", "index": tid}, variant="default", size="md",
                                       color="red"), label="Remove"),
        ], gap=4, wrap="nowrap"),
    ], justify="space-between", wrap="nowrap"), p=6)


def _metrics_table(views: list[TraceView]):
    from ...science.metrics import trace_metrics

    rows = []
    for k, tv in enumerate(views):
        t = tv.trace
        if t.x_unit != "s":
            continue
        m = trace_metrics(t.y, t.t0, t.dt)
        u = t.y_unit
        rows.append([f"[{k + 1}]", html.Code(tv.label[:48]), si_format(m.peak, u), f"{m.peak_time:.4g}",
                     si_format(m.peak_to_peak, u), si_format(m.rms, u),
                     f"{m.dominant_freq:.3g}" if m.dominant_freq else "—",
                     f"{m.duration_5_95:.3g}" if m.duration_5_95 is not None else "—",
                     f"{m.energy:.3g}", f"{m.cav:.3g}"])
    if not rows:
        return dmc.Text("Metrics are computed for time-domain traces.", size="sm", c="dimmed")
    return dmc.Stack([
        simple_table(["", "Trace", "Peak |x|", "t peak (s)", "Peak-to-peak", "RMS", "f dom (Hz)", "D5–95 (s)",
                      "∫x² dt", "∫|x| dt"], rows, fz="sm"),
        dmc.Text("D5–95: significant duration between 5 % and 95 % of ∫x² dt (Husid). For velocity traces the peak "
                 "is PGV; for acceleration, PGA.", size="xs", c="dimmed"),
    ])
