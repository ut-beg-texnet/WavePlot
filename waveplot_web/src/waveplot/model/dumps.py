"""Dump series: full-state dumps (velocities + stress tensor) on one plane over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..io.binsource import BinSource, read_dump
from ..io.records import DUMP_PLANE_COMPONENTS, DumpRecord
from .coords import AxisCoords, dump_axes, vertical_up

if TYPE_CHECKING:
    import xarray as xr


def dump_key(rec: DumpRecord) -> tuple:
    return (rec.axes, rec.extent.as_tuple(), rec.iqty, rec.jqty, rec.kqty, rec.vqty, rec.sx, rec.sy, rec.sz)


@dataclass
class DumpSeries:
    slug: str
    records: list[DumpRecord]
    source: BinSource

    @property
    def first(self) -> DumpRecord:
        return self.records[0]

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
    def dim3(self) -> bool:
        return self.first.dim3

    @property
    def vertical_up(self) -> bool:
        return vertical_up(self.ax2)

    @property
    def shape(self) -> tuple[int, int]:
        return self.first.xqty, self.first.yqty

    @property
    def label(self) -> str:
        r = self.first
        where = f" {r.fixed_axis}={r.fixed_index}" if r.fixed_axis else ""
        kind = "3-D tensor" if r.dim3 else "2-D"
        return f"[{r.axes}{where}] {r.xqty}×{r.yqty}, {kind}, {len(self.records)} dump(s)"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def times(self) -> np.ndarray:
        return np.array([r.time for r in self.records], dtype=float)

    def coords(self) -> tuple[AxisCoords, AxisCoords]:
        return dump_axes(self.first)

    def raw(self, n: int) -> np.ndarray:
        """All components of dump ``n`` as ``(xqty, yqty, vqty)`` float32."""
        return read_dump(self.source, self.records[n])

    def components(self, n: int) -> dict[str, np.ndarray]:
        """All components keyed by name (``v1 v2 [v3] s11 s22 [s33] s12 [s23 s13]``)."""
        arr = self.raw(n)
        return {name: arr[..., k] for k, name in enumerate(self.records[n].components)}

    def in_plane(self, n: int) -> dict[str, np.ndarray]:
        """In-plane velocity and stress components: keys ``u, v, s11, s22, s12`` (plane-relative)."""
        comps = self.components(n)
        rec = self.records[n]
        if not rec.dim3:
            names = ("v1", "v2", "s11", "s22", "s12")
        else:
            names = DUMP_PLANE_COMPONENTS.get(rec.axes, DUMP_PLANE_COMPONENTS["ij"])
        keys = ("u", "v", "s11", "s22", "s12")
        return {k: comps[nm] for k, nm in zip(keys, names, strict=True)}

    def to_xarray(self, n: int) -> xr.Dataset:
        import xarray as xr

        a1, a2 = self.coords()
        d1, d2 = a1.label, a2.label
        rec = self.records[n]
        comps = self.components(n)
        units = {c: ("m/s" if c.startswith("v") else "Pa") for c in comps}
        return xr.Dataset(
            {c: ((d1, d2), v, {"units": units[c]}) for c, v in comps.items()},
            coords={
                d1: (d1, a1.coord, {"units": "m", "grid_axis": a1.axis}),
                d2: (d2, a2.coord, {"units": "m", "grid_axis": a2.axis}),
                f"{a1.axis}_index": (d1, a1.index),
                f"{a2.axis}_index": (d2, a2.index),
            },
            attrs={"time": rec.time, "dump_number": rec.number, "axes": rec.axes,
                   "fixed_axis": rec.fixed_axis or "",
                   "fixed_index": rec.fixed_index if rec.fixed_index is not None else -1,
                   "series": self.slug},
        )


def build_dump_series(dumps: list[DumpRecord], source: BinSource) -> dict[str, DumpSeries]:
    groups: dict[tuple, list[DumpRecord]] = {}
    for rec in dumps:
        groups.setdefault(dump_key(rec), []).append(rec)
    out: dict[str, DumpSeries] = {}
    for recs in groups.values():
        recs.sort(key=lambda r: (r.time, r.index))
        r = recs[0]
        slug = f"{r.axes}|{r.fixed_axis}={r.fixed_index}" if r.fixed_axis else r.axes
        base, n = slug, 2
        while slug in out:
            slug = f"{base}#{n}"
            n += 1
        out[slug] = DumpSeries(slug, recs, source)
    return out
