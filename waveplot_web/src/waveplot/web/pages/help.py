"""Help page: user documentation, keyboard shortcuts, generated operation reference and the
table of v407 differences."""

from __future__ import annotations

from pathlib import Path

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, dcc

from ...formula.ops import ops_by_category
from ...science.compat import QUIRKS

DOCS = Path(__file__).resolve().parents[2] / "docs"
TOPICS = [("overview", "Overview"), ("shortcuts", "Keyboard shortcuts"), ("formulas", "Formulas"),
          ("operations", "Operation reference"), ("v407", "v407 differences"), ("formats", "File formats")]


def layout(topic: str | None = None, **_kwargs):
    return dmc.Grid([
        dmc.GridCol(dmc.Paper(dmc.Stack([
            dmc.NavLink(label=label, id={"type": "help-nav", "topic": key}, href=f"/help?topic={key}",
                        active=(topic or "overview") == key) for key, label in TOPICS], gap=2)),
            span={"base": 12, "md": 2}),
        dmc.GridCol(dmc.Paper(dcc.Markdown(id="help-body", mathjax=True, link_target="_blank",
                                           children=_content(topic or "overview"), className="wp-help")),
                    span={"base": 12, "md": 10}),
    ], gutter="sm")


def _read(name: str) -> str:
    try:
        return (DOCS / f"{name}.md").read_text(encoding="utf-8")
    except OSError:
        return f"*{name}.md not found*"


def _operations() -> str:
    lines = ["# Operation reference", "",
             "Generated from the operation registry. Parameters in parentheses are optional; units in brackets.", ""]
    for cat, specs in ops_by_category().items():
        lines.append(f"## {cat.capitalize()}")
        lines.append("")
        lines.append("| operation | inputs | parameters | v407 | description |")
        lines.append("|---|---|---|---|---|")
        for s in specs:
            params = ", ".join(f"{p.name}={p.default!r}" + (f" [{p.unit}]" if p.unit else "") for p in s.params)
            inputs = {0: "histories (h)", 1: "1 trace", 2: "2 traces"}.get(s.arity, str(s.arity))
            legacy = f"`{s.legacy}`" if s.legacy else ""
            doc = (s.doc or s.title).replace("|", "\\|")
            lines.append(f"| `{s.name}` | {inputs} | {params.replace('|', '/')} | {legacy.replace('|', '/')} | "
                         f"{doc} |")
        lines.append("")
    return "\n".join(lines)


def _v407() -> str:
    lines = ["# Differences between Standard and v407 numerics", "",
             "With **v407** selected in the header, WavePlot reproduces the original program's calculations. "
             "These are the behaviours that differ from **Standard** mode (the badge on each page lists the "
             "ones that affected the current result):", "", "| key | v407 behaviour |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in QUIRKS.items()]
    lines += ["", "Limits of v407 that are **not** reproduced in either mode: at most 8 traces, 250 snapshots, "
              "50 dumps (later dumps overwrote the last one), 4096-sample histories (only truncated in v407 mode "
              "for formula evaluation), a 101×101 display grid, 24 contour lines and 6×6 panels."]
    return "\n".join(lines)


def _shortcuts() -> str:
    from ..shell import SHORTCUTS

    rows = "\n".join(f"| `{k}` | {v} |" for k, v in SHORTCUTS)
    return ("# Keyboard shortcuts\n\nShortcuts work on the Snapshots page (arrows, Home/End and Space also on "
            "Dumps) whenever the focus is not in a text field. They follow the WavePlot v407 mnemonics.\n\n"
            f"| key | action |\n|---|---|\n{rows}\n")


def _formats() -> str:
    dev = Path(__file__).resolve().parents[4] / "docs" / "dev" / "formats.md"
    try:
        return dev.read_text(encoding="utf-8")
    except OSError:
        return "# File formats\n\nSee `docs/dev/formats.md` in the source distribution."


def _content(topic: str) -> str:
    if topic == "operations":
        return _operations()
    if topic == "v407":
        return _v407()
    if topic == "shortcuts":
        return _shortcuts()
    if topic == "formats":
        return _formats()
    return _read(topic if topic in ("overview", "formulas") else "overview")


def register_callbacks(app) -> None:
    @app.callback(Output("help-body", "children"), Output({"type": "help-nav", "topic": ALL}, "active"),
                  Input("url", "search"), State({"type": "help-nav", "topic": ALL}, "id"))
    def _topic(search, ids):
        from urllib.parse import parse_qs

        q = parse_qs((search or "").lstrip("?"))
        topic = (q.get("topic") or ["overview"])[0]
        return _content(topic), [i["topic"] == topic for i in ids]
