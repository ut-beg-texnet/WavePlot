"""Building :class:`PanelData` objects from datasets (shared by web, CLI exports and notebooks)."""

from __future__ import annotations

import numpy as np

from ..model.snapshots import SnapshotSeries
from .overlays import PlaneSpec
from .snapshot_fig import PanelData


def plane_spec(ds, series: SnapshotSeries) -> PlaneSpec | None:
    r = series.first
    if r.is_spacetime:
        return None
    sp = dict(ds.spacing)
    sp[r.ax1] = r.dx or sp.get(r.ax1, 1.0)
    sp[r.ax2] = r.dy or sp.get(r.ax2, 1.0)
    return PlaneSpec(r.ax1, r.ax2, r.fixed_axis, r.fixed_index, sp)


def frame_title(series: SnapshotSeries, n: int) -> str:
    r = series.records[n]
    where = f", {r.fixed_axis} = {r.fixed_index}" if r.fixed_axis else ""
    return f"{r.name} [{r.axes}{where}]  t = {r.time:.5g} s  (#{r.number}, frame {n + 1}/{len(series)})"


def snapshot_panel(ds, series: SnapshotSeries, n: int, data: np.ndarray | None = None,
                   title: str | None = None) -> PanelData:
    a1, a2 = series.coords()
    z = series.frame(n) if data is None else data
    return PanelData(
        z=z, x=a1.coord, y=a2.coord, ax1=a1.axis, ax2=a2.axis,
        title=title or frame_title(series, n), units=series.units, label=series.var_name,
        spacetime=series.is_spacetime, vertical_up=series.vertical_up, plane=plane_spec(ds, series),
        series_range=series.global_range(),
        extra={"time": float(series.records[n].time), "number": series.records[n].number},
    )
