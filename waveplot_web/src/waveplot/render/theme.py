"""Plotly templates matching the app's light and dark themes."""

from __future__ import annotations

import copy
import functools

import plotly.graph_objects as go
import plotly.io as pio

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, 'SFMono-Regular', Menlo, monospace"

PALETTE = ["#1c7ed6", "#e8590c", "#2f9e44", "#c2255c", "#7048e8", "#0c8599", "#f08c00", "#5c940d",
           "#ae3ec9", "#495057"]

GEOMETRY_COLORS = {
    "light": {"material": "#495057", "source": "#0b7285", "stope": "#c92a2a", "receiver": "#1c1c1c",
              "outline": "#212529", "zero": "#ffffff"},
    "dark": {"material": "#ced4da", "source": "#3bc9db", "stope": "#ff6b6b", "receiver": "#f8f9fa",
             "outline": "#f1f3f5", "zero": "#1a1b1e"},
}


@functools.lru_cache(maxsize=2)
def template(scheme: str = "light") -> go.layout.Template:
    base = copy.deepcopy(pio.templates["plotly_dark" if scheme == "dark" else "plotly_white"])
    bg = "#1a1b1e" if scheme == "dark" else "#ffffff"
    paper = "#1f2023" if scheme == "dark" else "#ffffff"
    grid = "#2c2e33" if scheme == "dark" else "#e9ecef"
    text = "#e9ecef" if scheme == "dark" else "#212529"
    base.layout.update(
        font=dict(family=FONT, size=12, color=text),
        paper_bgcolor=paper,
        plot_bgcolor=bg,
        colorway=PALETTE,
        margin=dict(l=60, r=20, t=40, b=50),
        hoverlabel=dict(font=dict(family=MONO, size=11)),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid, ticks="outside", showline=True, mirror=True),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid, ticks="outside", showline=True, mirror=True),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return base


def geometry_colors(scheme: str) -> dict[str, str]:
    return GEOMETRY_COLORS["dark" if scheme == "dark" else "light"]


def empty_figure(message: str, scheme: str = "light") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=template(scheme), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                                        font=dict(size=14))],
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def si_format(v: float, unit: str = "", digits: int = 4) -> str:
    """Engineering-prefixed number: 2.5e6 Pa → '2.5 MPa'."""
    import math

    if v == 0 or not math.isfinite(v):
        return f"{v:g} {unit}".strip()
    prefixes = {-12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
    exp = int(math.floor(math.log10(abs(v)) / 3) * 3)
    exp = max(-12, min(12, exp))
    if not unit:
        return f"{v:.{digits}g}"
    return f"{v / 10**exp:.{digits}g} {prefixes[exp]}{unit}"
