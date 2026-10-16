"""Reusable UI building blocks."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html


def section(title: str, right=None):
    if right is None:
        return html.Div(title, className="wp-section-title")
    return dmc.Group([html.Div(title, className="wp-section-title"), right], justify="space-between")


def kv_table(rows: list[tuple[str, object]]):
    return dmc.Table(dmc.TableTbody([dmc.TableTr([dmc.TableTd(k), dmc.TableTd(v)]) for k, v in rows]),
                     className="wp-kv", verticalSpacing=2, fz="sm")


def simple_table(headers: list[str], rows: list[list], striped: bool = True, fz: str = "xs"):
    return dmc.Table([
        dmc.TableThead(dmc.TableTr([dmc.TableTh(h) for h in headers])),
        dmc.TableTbody([dmc.TableTr([dmc.TableTd(c) for c in r]) for r in rows]),
    ], striped=striped, highlightOnHover=True, withTableBorder=False, fz=fz, stickyHeader=True,
        verticalSpacing=3)


def provenance_badge(quirks: list[str], mode: str):
    from ...science.compat import QUIRKS

    if mode != "v407":
        return dmc.Badge("standard numerics", color="teal", variant="light", size="sm", tt="none")
    if not quirks:
        return dmc.Badge("v407 mode", color="orange", variant="light", size="sm", tt="none")
    return dmc.Tooltip(dmc.Badge(f"v407 numerics · {len(quirks)} quirk(s)", color="orange", variant="filled",
                                 size="sm", tt="none"),
                       label=html.Ul([html.Li(QUIRKS.get(q, q)) for q in quirks], style={"margin": 0,
                                                                                        "paddingLeft": 16}),
                       multiline=True, w=360, withArrow=True)


def labeled(label: str, comp, tip: str | None = None):
    lab = dmc.Text(label, size="xs", fw=500)
    if tip:
        lab = dmc.Tooltip(lab, label=tip, multiline=True, w=260, withArrow=True)
    return dmc.Stack([lab, comp], gap=2)
