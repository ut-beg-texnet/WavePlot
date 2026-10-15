"""Snapshot series: snapshots of the same variable on the same plane, ordered in time."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..io.binsource import BinSource, read_snapshot, read_snapshot_rows
from ..io.names import FUNC_DESCRIPTIONS, func_name
from ..io.records import SnapRecord
from .coords import AxisCoords, snapshot_axes, vertical_up

if TYPE_CHECKING:
    import xarray as xr


def series_key(rec: SnapRecord) -> tuple:
    return (rec.var_raw, rec.ax1, rec.ax2, rec.extent.as_tuple(), rec.sx, rec.sy, rec.xqty, rec.yqty)


def _slug(rec: SnapRecord) -> str:
    name = rec.name.replace(" ", "")
    if rec.is_spacetime:
        return f"{name}|{rec.axes}"
    fa = rec.fixed_axis
    return f"{name}|{rec.axes}|{fa}={rec.fixed_index}" if fa else f"{name}|{rec.axes}"


@dataclass
class SnapshotSeries:
    slug: str
    records: list[SnapRecord]
    source: BinSource
    stope_id: int | None = None
    _reader: Callable[[SnapRecord], np.ndarray] | None = field(default=None, repr=False)

    # --- identity -------------------------------------------------------------------------
    @property
    def first(self) -> SnapRecord:
        return self.records[0]

    @property
    def name(self) -> str:
        return self.first.name

    @property
    def var_name(self) -> str:
        return self.first.info.name

    @property
    def long_name(self) -> str:
        r = self.first
        desc = r.info.long_name
        f = func_name(r.func)
        return f"{desc} ({FUNC_DESCRIPTIONS.get(f, f)})" if f else desc

    @property
    def units(self) -> str:
        return self.first.units

    @property
    def axes(self) -> str:
        return self.first.axes

    @property
    def ax1(self) -> str:
        return self.first.ax1

    @property
    def ax2(self) -> str:
        return self.first.ax2

    @property
    def is_spacetime(self) -> bool:
        return self.first.is_spacetime

    @property
    def is_stope(self) -> bool:
        return self.first.info.is_stope

    @property
    def shape(self) -> tuple[int, int]:
        return self.first.xqty, self.first.yqty

    @property
    def label(self) -> str:
        r = self.first
        where = f" {r.fixed_axis}={r.fixed_index}" if r.fixed_axis else ""
        return f"{r.name} [{r.axes}{where}] {r.xqty}×{r.yqty}, {len(self.records)} frame(s)"

    @property
    def vertical_up(self) -> bool:
        return vertical_up(self.ax2)

    # --- frames -----------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    @property
    def times(self) -> np.ndarray:
        return np.array([r.time for r in self.records], dtype=float)

    def coords(self) -> tuple[AxisCoords, AxisCoords]:
        return snapshot_axes(self.first)

    def frame(self, n: int) -> np.ndarray:
        """Frame ``n`` (0-based) as ``(xqty, yqty)`` float32 (rows along ``ax1``)."""
        rec = self.records[n]
        return self._reader(rec) if self._reader else read_snapshot(self.source, rec)

    def frame_index_for_time(self, t: float) -> int:
        return int(np.abs(self.times - t).argmin())

    def cell_series(self, ix: int, iy: int, frames: Sequence[int] | None = None) -> np.ndarray:
        """Value of one cell across frames (reads one row per frame only)."""
        idx = range(len(self.records)) if frames is None else frames
        out = np.empty(len(idx), dtype=np.float32)
        for n, f in enumerate(idx):
            out[n] = read_snapshot_rows(self.source, self.records[f], ix, ix + 1)[0, iy]
        return out

    def global_range(self) -> tuple[float, float]:
        """Min/max over all frames from the ``.map`` header (no data read)."""
        return (min(r.minval for r in self.records), max(r.maxval for r in self.records))

    def envelope(self, kind: str = "absmax", frames: Sequence[int] | None = None) -> np.ndarray:
        """Per-cell maximum / minimum / signed abs-max over frames (v407 ``DI_ExtractSnap``)."""
        idx = list(range(len(self.records)) if frames is None else frames)
        acc: np.ndarray | None = None
        for f in idx:
            a = self.frame(f).astype(np.float32)
            if acc is None:
                acc = a.copy()
            elif kind == "max":
                np.maximum(acc, a, out=acc)
            elif kind == "min":
                np.minimum(acc, a, out=acc)
            else:
                acc = np.where(np.abs(a) > np.abs(acc), a, acc)
        if acc is None:
            raise ValueError("no frames")
        return acc

    def to_xarray(self, frames: Sequence[int] | None = None) -> xr.DataArray:
        import xarray as xr

        idx = list(range(len(self.records)) if frames is None else frames)
        a1, a2 = self.coords()
        data = np.stack([self.frame(f) for f in idx]) if idx else np.empty((0, *self.shape), np.float32)
        d1, d2 = a1.label, a2.label
        r = self.first
        return xr.DataArray(
            data,
            dims=("time", d1, d2),
            coords={
                "time": ("time", self.times[idx], {"units": "s"}),
                d1: (d1, a1.coord, {"units": a1.unit, "grid_axis": a1.axis}),
                d2: (d2, a2.coord, {"units": a2.unit, "grid_axis": a2.axis}),
                f"{a1.axis}_index" if a1.axis != "t" else "cycle_1": (d1, a1.index),
                f"{a2.axis}_index" if a2.axis != "t" else "cycle_2": (d2, a2.index),
                "snapshot_number": ("time", [self.records[f].number for f in idx]),
            },
            name=self.var_name,
            attrs={
                "long_name": self.long_name,
                "units": self.units,
                "wave_var": r.var,
                "wave_func": func_name(r.func),
                "axes": r.axes,
                "fixed_axis": r.fixed_axis or "",
                "fixed_index": r.fixed_index if r.fixed_index is not None else -1,
                "stagger": list(r.stagger),
                "series": self.slug,
            },
        )


def build_series(snaps: list[SnapRecord], source: BinSource) -> dict[str, SnapshotSeries]:
    groups: dict[tuple, list[SnapRecord]] = {}
    for rec in snaps:
        groups.setdefault(series_key(rec), []).append(rec)
    out: dict[str, SnapshotSeries] = {}
    for recs in groups.values():
        recs.sort(key=lambda r: (r.time, r.index))
        slug = _slug(recs[0])
        if slug in out:
            e = recs[0].extent
            slug = f"{slug}|{e.i1}-{e.i2},{e.j1}-{e.j2},{e.k1}-{e.k2}"
            n = 2
            base = slug
            while slug in out:
                slug = f"{base}#{n}"
                n += 1
        out[slug] = SnapshotSeries(slug, recs, source)
    return out
