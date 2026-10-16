"""Application shell: header (dataset picker, numerics mode, theme), navigation, global stores."""

from __future__ import annotations

from urllib.parse import parse_qs

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback_context, dcc, html, no_update

from ..config import AppConfig
from . import state
from .app import PAGES, mantine_theme
from .prefs import load_preferences

SHORTCUTS = [
    ("← / →", "previous / next frame (Shift: jump by N)"),
    ("Home / End", "first / last frame"),
    ("Space", "play / pause animation"),
    ("+ / −", "faster / slower animation"),
    ("M / C / L", "colour map / contours / multi-line view"),
    ("A", "toggle absolute-value colouring"),
    ("B", "toggle banded (discrete level) colours"),
    ("> / <", "more / fewer colour levels"),
    ("D", "cycle level distribution"),
    ("K", "keep (lock) the current colour range"),
    ("I", "toggle smooth (interpolated) colour map"),
    ("G", "toggle geometry overlays"),
    ("H / V", "multi-line direction horizontal / vertical"),
    ("X / x", "bigger / smaller line or vector scale"),
    ("?", "show this list"),
]


def _nav_links() -> list:
    return [dmc.NavLink(label=name, href=path, id=f"nav-{mod}", leftSection=html.Span(icon, className="wp-nav-icon"),
                        description=None, active=False)
            for mod, path, name, icon, _desc in PAGES]


def layout(cfg: AppConfig):
    header = dmc.Group([
        dmc.Group([
            dmc.Burger(id="nav-burger", size="sm", hiddenFrom="sm", opened=False),
            html.Div([html.Span("Wave", className="wp-logo-a"), html.Span("Plot", className="wp-logo-b")],
                     className="wp-logo"),
            dmc.Badge("web", size="xs", variant="light", color="gray"),
        ], gap="xs"),
        dmc.Group([
            dmc.Select(id="ds-select", placeholder="Choose a dataset…", searchable=True, clearable=False,
                       w=340, size="sm", data=[], nothingFoundMessage="No .map files under the data roots",
                       comboboxProps={"withinPortal": True}),
            dmc.Tooltip(
                dmc.SegmentedControl(id="mode-switch", data=[{"label": "Standard", "value": "standard"},
                                                              {"label": "v407", "value": "v407"}],
                                     value="standard", size="xs", persistence=True, persistence_type="local"),
                label="Numerics: Standard = corrected, conventional maths. v407 = reproduce the original "
                      "WavePlot numerics (including its known quirks) for comparison with old results.",
                multiline=True, w=300, withArrow=True),
            dmc.Tooltip(dmc.ActionIcon("☾", id="theme-toggle", variant="default", size="lg"),
                        label="Toggle light / dark theme"),
            dmc.Tooltip(dmc.ActionIcon("?", id="shortcuts-open", variant="default", size="lg"),
                        label="Keyboard shortcuts"),
        ], gap="sm"),
    ], justify="space-between", h="100%", px="md")

    shortcuts = dmc.Modal(
        id="shortcuts-modal", title="Keyboard shortcuts", size="lg",
        children=dmc.Table([dmc.TableTbody([dmc.TableTr([dmc.TableTd(dmc.Kbd(k)), dmc.TableTd(v)])
                                            for k, v in SHORTCUTS])], striped=True))

    shell = dmc.AppShell(
        [
            dmc.AppShellHeader(header),
            dmc.AppShellNavbar([dmc.Stack(_nav_links(), gap=2), html.Div(id="nav-footer", className="wp-nav-footer")],
                               p="xs"),
            dmc.AppShellMain(dash.page_container, className="wp-main"),
        ],
        header={"height": 56},
        navbar={"width": 200, "breakpoint": "sm", "collapsed": {"mobile": True}},
        padding="sm",
        id="appshell",
    )
    return dmc.MantineProvider(
        [
            dcc.Location(id="url", refresh="callback-nav"),
            dcc.Store(id="store-ds", storage_type="session", data=cfg.initial_dataset),
            dcc.Store(id="store-prefs", storage_type="local", data=load_preferences().model_dump()),
            dcc.Store(id="store-kbd", data=None),
            dcc.Store(id="store-nav", storage_type="session", data=None),
            dmc.NotificationContainer(id="notify", position="bottom-right"),
            shortcuts,
            shell,
        ],
        id="provider", theme=mantine_theme(), forceColorScheme="light",
    )


def dataset_options() -> list[dict]:
    entries = state.registry().list_datasets()
    groups: dict[str, list[dict]] = {}
    for e in entries:
        size = e.size / 1024
        groups.setdefault(e.root, []).append({"value": e.id, "label": e.relative})
        _ = size
    return [{"group": root, "items": items} for root, items in groups.items()]


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("ds-select", "data"),
        Output("ds-select", "value"),
        Input("url", "search"),
        State("store-ds", "data"),
    )
    def _init_datasets(search, current):
        opts = dataset_options()
        ids = [i["value"] for g in opts for i in g["items"]]
        q = parse_qs((search or "").lstrip("?"))
        want = (q.get("ds") or [None])[0] or current
        if want not in ids:
            want = ids[0] if len(ids) == 1 else (want if want in ids else None)
        return opts, want

    @app.callback(Output("store-ds", "data"), Input("ds-select", "value"), prevent_initial_call=True)
    def _select_dataset(value):
        return value or no_update

    @app.callback(
        Output("store-prefs", "data"),
        Input("mode-switch", "value"),
        Input("theme-toggle", "n_clicks"),
        State("store-prefs", "data"),
        prevent_initial_call=True,
    )
    def _prefs(mode, _clicks, prefs):
        prefs = dict(prefs or {})
        trig = callback_context.triggered_id
        if trig == "mode-switch":
            prefs["mode"] = mode
        elif trig == "theme-toggle":
            prefs["scheme"] = "dark" if state.scheme_of(prefs) == "light" else "light"
        return prefs

    @app.callback(
        Output("provider", "forceColorScheme"),
        Output("theme-toggle", "children"),
        Input("store-prefs", "data"),
    )
    def _apply_prefs(prefs):
        scheme = state.scheme_of(prefs)
        return scheme, ("☀" if scheme == "dark" else "☾")

    @app.callback([Output(f"nav-{mod}", "active") for mod, *_ in PAGES], Input("url", "pathname"))
    def _active(path):
        path = path or "/"
        return [(p == "/" and path == "/") or (p != "/" and path.startswith(p)) for _m, p, *_ in PAGES]

    @app.callback(Output("shortcuts-modal", "opened"), Input("shortcuts-open", "n_clicks"),
                  Input("store-kbd", "data"), State("shortcuts-modal", "opened"), prevent_initial_call=True)
    def _shortcuts(_n, kbd, opened):
        if callback_context.triggered_id == "store-kbd":
            if kbd and kbd.get("key") == "?":
                return not opened
            return no_update
        return not opened

    @app.callback(Output("appshell", "navbar"), Input("nav-burger", "opened"), State("appshell", "navbar"))
    def _burger(opened, navbar):
        navbar = dict(navbar or {})
        navbar["collapsed"] = {"mobile": not opened}
        return navbar

    @app.callback(Output("nav-footer", "children"), Input("store-ds", "data"))
    def _footer(ds_id):
        ds = state.get_dataset(ds_id)
        if ds is None:
            return dmc.Text("No dataset selected", size="xs", c="dimmed")
        n_err = sum(1 for i in ds.issues if i.severity == "error")
        return dmc.Stack([
            dmc.Text(ds.name, size="xs", fw=600),
            dmc.Text(f"{len(ds.series)} snapshot series · {len(ds.histories)} histories · {len(ds.dumps)} dumps",
                     size="xs", c="dimmed"),
            dmc.Badge(f"{n_err} file issue(s)", color="red", size="xs") if n_err else None,
        ], gap=2)
