"""Dash application factory."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_mantine_components as dmc
from dash import Dash

from ..config import AppConfig
from . import state

ASSETS = Path(__file__).parent / "assets"

PAGES = [
    # (module, path, name, icon glyph, description)
    ("datasets", "/", "Datasets", "▤", "Browse runs, summaries and metadata"),
    ("snapshots", "/snapshots", "Snapshots", "▦", "2-D field planes over time"),
    ("dumps", "/dumps", "Dumps", "✣", "Stress tensors and velocities on a plane"),
    ("histories", "/histories", "Histories", "∿", "Receiver time series and processing"),
    ("geometry", "/geometry", "Geometry", "◫", "Model geometry, 3-D view and particle motion"),
    ("compare", "/compare", "Compare", "⇄", "Compare runs side by side or as differences"),
    ("settings", "/settings", "Settings", "⚙", "Preferences and legacy settings import"),
    ("help", "/help", "Help", "?", "Documentation, shortcuts, formula reference"),
]


def create_app(cfg: AppConfig) -> Dash:
    state.init(cfg)
    app = Dash(
        __name__,
        title=cfg.title,
        use_pages=True,
        pages_folder="",
        assets_folder=str(ASSETS),
        suppress_callback_exceptions=True,
        serve_locally=True,
        update_title=None,
    )
    import importlib

    from .shell import layout as shell_layout
    from .shell import register_callbacks as shell_callbacks

    for order, (mod_name, path, name, _icon, desc) in enumerate(PAGES):
        mod = importlib.import_module(f".pages.{mod_name}", __package__)
        dash.register_page(f"waveplot.{mod_name}", path=path, name=name, order=order, title=f"{name} · WavePlot",
                           description=desc, layout=mod.layout)
        mod.register_callbacks(app)
    app.layout = shell_layout(cfg)
    shell_callbacks(app)

    from .routes import register_routes

    register_routes(app)
    return app


def mantine_theme() -> dict:
    return {
        "primaryColor": "indigo",
        "defaultRadius": "md",
        "fontFamily": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "fontFamilyMonospace": "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace",
        "components": {
            "Paper": {"defaultProps": {"withBorder": True, "p": "sm"}},
            "Select": {"defaultProps": {"size": "xs"}},
            "NumberInput": {"defaultProps": {"size": "xs"}},
            "TextInput": {"defaultProps": {"size": "xs"}},
            "SegmentedControl": {"defaultProps": {"size": "xs"}},
            "Switch": {"defaultProps": {"size": "xs"}},
            "Checkbox": {"defaultProps": {"size": "xs"}},
        },
    }


_ = dmc  # re-exported for type checkers
