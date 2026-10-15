"""``WaveDataset``: one WAVE run (``.map`` plus companion files), lazily loaded."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..io.binsource import BinSource, read_history
from ..io.crk import CrackElements, read_crack_dump
from ..io.discover import DatasetFiles, resolve_dataset
from ..io.geo import Geometry, read_geometry
from ..io.mapfile import read_map
from ..io.records import FormatIssue, HistRecord, SnapRecord
from ..io.script import RunLog, WaveScript, parse_log, parse_outline, parse_script
from .coords import history_axis
from .dumps import DumpSeries, build_dump_series
from .histories import Receiver, build_receivers
from .snapshots import SnapshotSeries, build_series

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


class WaveDataset:
    """A WAVE simulation output set.

    >>> ds = open_dataset("Waveplot_exe/mt_slip5.map")
    >>> ds.series["S11|ik|j=50"].frame(10).shape
    (510, 510)
    """

    def __init__(self, files: DatasetFiles, *, func_base: int = 0xFFFF, keep_files_open: bool | None = None):
        self.files = files
        self.name = files.stem
        self.index = read_map(files.map, func_base=func_base)
        self.issues: list[FormatIssue] = list(self.index.issues)
        self.snp = BinSource(files.get("snp"), keep_files_open)
        self.hst = BinSource(files.get("hst"), keep_files_open)
        self.dmp = BinSource(files.get("dmp"), keep_files_open)
        self._check_sizes()
        self.series: dict[str, SnapshotSeries] = build_series(self.index.snaps, self.snp)
        self.dump_series: dict[str, DumpSeries] = build_dump_series(self.index.dumps, self.dmp)
        self.receivers: list[Receiver] = build_receivers(self.index.hists)
        self._link_stopes()

    # --- basic collections -----------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.files.map

    @property
    def snapshots(self) -> list[SnapRecord]:
        return self.index.snaps

    @property
    def histories(self) -> list[HistRecord]:
        return self.index.hists

    @property
    def dumps(self):
        return self.index.dumps

    @property
    def geometries(self):
        return self.index.geoms

    @property
    def cracks(self):
        return self.index.cracks

    def __repr__(self) -> str:
        return (f"<WaveDataset {self.name}: {len(self.snapshots)} snapshots in {len(self.series)} series, "
                f"{len(self.histories)} histories, {len(self.dumps)} dumps, {len(self.geometries)} geometries>")

    def _check_sizes(self) -> None:
        for ext, expected in self.index.expected_sizes.items():
            if expected == 0:
                continue
            p = self.files.get(ext)
            if p is None:
                self.issues.append(FormatIssue(ext, f".{ext} file missing ({expected} bytes indexed)", "error"))
                continue
            size = p.stat().st_size
            if size < expected:
                self.issues.append(FormatIssue(
                    p.name, f"file has {size} bytes but the map indexes {expected}; later records unavailable",
                    "error"))
            elif size > expected:
                self.issues.append(FormatIssue(p.name, f"{size - expected} bytes beyond indexed data"))

    # --- geometry --------------------------------------------------------------------------
    @functools.cached_property
    def _geo_bytes(self) -> bytes:
        p = self.files.get("geo")
        return p.read_bytes() if p else b""

    @functools.cached_property
    def _outline(self) -> list[tuple[int, int]] | None:
        p = self.files.get("otl")
        return parse_outline(p) if p else None

    @functools.lru_cache(maxsize=8)  # noqa: B019 - dataset objects are long-lived and few
    def geometry_number(self, number: int) -> Geometry | None:
        for g in self.geometries:
            if g.number == number:
                geom = read_geometry(self._geo_bytes, g, self._outline)
                self.issues.extend(geom.issues)
                return geom
        return None

    @property
    def geometry(self) -> Geometry | None:
        """The (last) geometry of the run, or ``None``."""
        if not self.geometries or not self._geo_bytes:
            return None
        return self.geometry_number(self.geometries[-1].number)

    def geometry_for(self, gnum: int, gid: int) -> Geometry | None:
        for g in self.geometries:
            if g.gnum == gnum and g.gid == gid:
                return self.geometry_number(g.number)
        return self.geometry

    @property
    def spacing(self) -> dict[str, float]:
        if self.geometries:
            return self.geometries[-1].spacing
        if self.snapshots:
            s = self.snapshots[0]
            return {"i": s.dx, "j": s.dx, "k": s.dx}
        return {"i": 1.0, "j": 1.0, "k": 1.0}

    def _link_stopes(self) -> None:
        if not any(s.is_stope for s in self.series.values()):
            return
        try:
            geom = self.geometry
        except Exception:
            return
        if geom is None:
            return
        for s in self.series.values():
            if not s.is_stope or s.is_spacetime:
                continue
            r = s.first
            for st in geom.stopes:
                if st.plane == r.axes or (st.plane and set(st.plane) == set(r.axes)):
                    fa = r.fixed_axis
                    if fa and abs(st.extent.range(fa)[0] - (r.fixed_index or 0)) <= 1:
                        s.stope_id = st.stope_id
                        break

    # --- side files --------------------------------------------------------------------------
    @functools.cached_property
    def script(self) -> WaveScript | None:
        p = self.files.get("m")
        try:
            return parse_script(p) if p else None
        except OSError:
            return None

    @functools.cached_property
    def runlog(self) -> RunLog | None:
        p = self.files.get("log")
        try:
            return parse_log(p) if p else None
        except OSError:
            return None

    # --- data access ---------------------------------------------------------------------------
    def snapshot_record(self, number: int) -> SnapRecord:
        return self.snapshots[number - 1]

    def history_record(self, number: int) -> HistRecord:
        return self.histories[number - 1]

    def series_of(self, rec: SnapRecord) -> tuple[SnapshotSeries, int]:
        for s in self.series.values():
            for n, r in enumerate(s.records):
                if r.index == rec.index:
                    return s, n
        raise KeyError(rec.number)

    def history_data(self, number: int) -> np.ndarray:
        return read_history(self.hst, self.history_record(number))

    def snapshot(self, number: int) -> xr.DataArray:
        """Snapshot ``number`` (1-based, as listed by v407) as an ``xarray.DataArray``."""
        s, n = self.series_of(self.snapshot_record(number))
        return s.to_xarray([n]).isel(time=0)

    def history(self, number: int) -> xr.DataArray:
        import xarray as xr

        rec = self.history_record(number)
        dim = "distance" if rec.is_profile else "time"
        return xr.DataArray(
            self.history_data(number), dims=(dim,),
            coords={dim: (dim, history_axis(rec), {"units": rec.x_unit})},
            name=rec.info.name,
            attrs={"long_name": rec.info.long_name, "units": rec.units, "history_number": number,
                   "position_ijk": list(rec.position), "wave_var": rec.var},
        )

    def dump(self, number: int) -> xr.Dataset:
        rec = self.dumps[number - 1]
        for s in self.dump_series.values():
            for n, r in enumerate(s.records):
                if r.index == rec.index:
                    return s.to_xarray(n)
        raise KeyError(number)

    def crack_dump(self, number: int) -> list[CrackElements]:
        p = self.files.get("crk")
        if p is None:
            return []
        rec = self.cracks[number - 1]
        elems, issues = read_crack_dump(p.read_bytes(), rec)
        self.issues.extend(issues)
        return elems

    # --- summaries -------------------------------------------------------------------------------
    def summary(self) -> dict[str, pd.DataFrame]:
        import pandas as pd

        snaps = pd.DataFrame([{
            "number": r.number, "time": r.time, "variable": r.name, "units": r.units, "axes": r.axes,
            "fixed": f"{r.fixed_axis}={r.fixed_index}" if r.fixed_axis else "", "xqty": r.xqty,
            "yqty": r.yqty, "max": r.maxval, "min": r.minval, "geometry": r.gnum} for r in self.snapshots])
        series = pd.DataFrame([{
            "series": s.slug, "variable": s.name, "axes": s.axes, "shape": f"{s.shape[0]}×{s.shape[1]}",
            "frames": len(s), "t_first": s.times[0], "t_last": s.times[-1], "min": s.global_range()[0],
            "max": s.global_range()[1], "stope": s.stope_id} for s in self.series.values()])
        hists = pd.DataFrame([{
            "number": h.number, "variable": h.name, "units": h.units, "steps": h.n, "t_start": h.tstart,
            "t_end": h.tend, "dt": h.dt, "i": h.position[0], "j": h.position[1], "k": h.position[2],
            "max": h.maxval, "min": h.minval} for h in self.histories])
        dumps = pd.DataFrame([{
            "number": d.number, "time": d.time, "plane": d.axes,
            "fixed": f"{d.fixed_axis}={d.fixed_index}" if d.fixed_axis else "",
            "i": f"{d.extent.i1}-{d.extent.i2}", "j": f"{d.extent.j1}-{d.extent.j2}",
            "k": f"{d.extent.k1}-{d.extent.k2}", "components": d.vqty} for d in self.dumps])
        return {"snapshots": snaps, "series": series, "histories": hists, "dumps": dumps}

    def info(self) -> dict[str, Any]:
        """JSON-friendly overview used by the CLI and the web app."""
        geom = None
        try:
            geom = self.geometry
        except Exception as exc:
            self.issues.append(FormatIssue("geo", f"could not read geometry: {exc}", "error"))
        sc, log = self.script, self.runlog
        grid = None
        if geom is not None and geom.grid is not None:
            e = geom.grid
            grid = [e.i2 - e.i1, e.j2 - e.j1, e.k2 - e.k1]
        elif sc is not None and sc.grid:
            grid = list(sc.grid)
        out: dict[str, Any] = {
            "name": self.name,
            "folder": str(self.files.folder),
            "files": {k: {"name": v.name, "size": v.stat().st_size} for k, v in sorted(self.files.files.items())},
            "counts": {"snapshots": len(self.snapshots), "series": len(self.series),
                       "histories": len(self.histories), "receivers": len(self.receivers),
                       "dumps": len(self.dumps), "dump_series": len(self.dump_series),
                       "geometries": len(self.geometries), "crack_dumps": len(self.cracks)},
            "grid_cells": grid,
            "spacing_m": self.spacing,
            "model3d": geom.model3d if geom else None,
            "timestep_s": log.timestep if log else None,
            "final_cycle": list(log.final_cycle) if log and log.final_cycle else None,
            "issues": [str(i) for i in self.issues],
        }
        if geom is not None:
            out["materials"] = [{"number": p.number, "K": p.bulk, "G": p.shear, "rho": p.density,
                                 "vp": round(p.vp, 3), "vs": round(p.vs, 3), "poisson": round(p.poisson, 4)}
                                for p in geom.props]
            out["sources"] = len(geom.sources)
            out["stopes"] = len(geom.stopes)
        return out

    def close(self) -> None:
        for src in (self.snp, self.hst, self.dmp):
            src.close()

    def __enter__(self) -> WaveDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_dataset(path: str | os.PathLike[str], *, func_base: int = 0xFFFF,
                 keep_files_open: bool | None = None) -> WaveDataset:
    """Open a WAVE dataset from its ``.map`` file (or base name / folder)."""
    return WaveDataset(resolve_dataset(path), func_base=func_base, keep_files_open=keep_files_open)
