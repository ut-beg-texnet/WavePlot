"""Dumps page (implemented in a later milestone)."""

from __future__ import annotations

import dash_mantine_components as dmc


def layout(**_kwargs):
    return dmc.Paper(dmc.Text("Dumps: coming soon.", c="dimmed"))


def register_callbacks(app) -> None:
    return None
