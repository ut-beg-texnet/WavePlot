"""xarray backend: ``xr.open_datatree("run.map", engine="wave")`` (lazy, memory-friendly).

Snapshot series become ``/snapshots/<series>`` groups with a lazily loaded ``(time, a, b)``
variable — frames are only read from disk when values are accessed. Histories and dumps are
exposed as ``/histories`` and ``/dumps/<plane>``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing


def group_name(slug: str) -> str:
    """netCDF/HDF5-safe group name for a series slug (``S11|ik|j=50`` → ``S11_ik_j50``)."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug.replace("=", "")).strip("_")
    return s or "series"


class _FramesArray(BackendArray):
    def __init__(self, n: int, shape2d: tuple[int, ...], reader, dtype=np.float32):
        self.shape = (n, *shape2d)
        self.dtype = np.dtype(dtype)
        self._reader = reader

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._raw)

    def _raw(self, key: tuple) -> np.ndarray:
        tkey, rest = key[0], key[1:]
        frames = np.arange(self.shape[0])[tkey]
        scalar = np.ndim(frames) == 0
        data = []
        for f in np.atleast_1d(frames):
            a = np.asarray(self._reader(int(f)))
            # outer indexing, last axis first so axis numbers stay valid; ints drop the axis
            for axis in range(len(rest) - 1, -1, -1):
                k = rest[axis]
                idx = k if isinstance(k, int | np.integer) else np.arange(a.shape[axis])[k]
                a = np.take(a, idx, axis=axis)
            data.append(a)
        if not data:
            return np.empty((0, *self.shape[1:]), self.dtype)
        out = np.stack(data)
        return out[0] if scalar else out


def series_dataset(series) -> xr.Dataset:
    a1, a2 = series.coords()
    d1, d2 = a1.label, a2.label
    arr = _FramesArray(len(series), series.shape, series.frame)
    var = xr.Variable(("time", d1, d2), indexing.LazilyIndexedArray(arr),
                      attrs={"units": series.units, "long_name": series.long_name})
    r = series.first
    return xr.Dataset(
        {series.var_name: var},
        coords={"time": ("time", series.times, {"units": "s"}),
                d1: (d1, a1.coord, {"units": a1.unit, "grid_axis": a1.axis}),
                d2: (d2, a2.coord, {"units": a2.unit, "grid_axis": a2.axis})},
        attrs={"series": series.slug, "axes": r.axes, "fixed_axis": r.fixed_axis or "",
               "fixed_index": r.fixed_index if r.fixed_index is not None else -1},
    )


def histories_dataset(ds) -> xr.Dataset | None:
    recs = [h for h in ds.histories if not h.is_profile]
    if not recs:
        return None
    n = max(h.n for h in recs)
    data = np.full((len(recs), n), np.nan, dtype=np.float32)
    t0 = np.array([h.tstart for h in recs])
    dt = np.array([h.dt for h in recs])
    for k, h in enumerate(recs):
        data[k, :h.n] = ds.history_data(h.number)
    pos = np.array([h.position for h in recs])
    return xr.Dataset(
        {"value": (("history", "sample"), data)},
        coords={"history": ("history", [h.number for h in recs]),
                "variable": ("history", [h.name for h in recs]),
                "units": ("history", [h.units for h in recs]),
                "i": ("history", pos[:, 0]), "j": ("history", pos[:, 1]), "k": ("history", pos[:, 2]),
                "t0": ("history", t0, {"units": "s"}), "dt": ("history", dt, {"units": "s"}),
                "sample": ("sample", np.arange(n))},
        attrs={"description": "time of sample n = t0 + n*dt"},
    )


def dump_dataset(series) -> xr.Dataset:
    a1, a2 = series.coords()
    d1, d2 = a1.label, a2.label
    comps = series.first.components
    data_vars = {}
    for c_idx, c in enumerate(comps):
        def reader(f, c_idx=c_idx):
            return series.raw(f)[..., c_idx]
        arr = _FramesArray(len(series), series.shape, reader)
        data_vars[c] = xr.Variable(("time", d1, d2), indexing.LazilyIndexedArray(arr),
                                   attrs={"units": "m/s" if c.startswith("v") else "Pa"})
    return xr.Dataset(data_vars,
                      coords={"time": ("time", series.times, {"units": "s"}),
                              d1: (d1, a1.coord, {"units": "m"}), d2: (d2, a2.coord, {"units": "m"})},
                      attrs={"series": series.slug, "axes": series.axes})


def geometry_dataset(ds) -> xr.Dataset | None:
    g = ds.geometry
    if g is None:
        return None
    mats = g.materials
    out = xr.Dataset(
        {"material_number": ("material", [m.number for m in mats]),
         "material_extent": (("material", "bound"), np.array([m.extent.as_tuple() for m in mats]).reshape(-1, 6))},
        coords={"bound": ["i1", "i2", "j1", "j2", "k1", "k2"]},
        attrs={"dx": g.spacing["i"], "dy": g.spacing["j"], "dz": g.spacing["k"]},
    )
    if g.props:
        out["bulk_modulus"] = ("property", [p.bulk for p in g.props], {"units": "Pa"})
        out["shear_modulus"] = ("property", [p.shear for p in g.props], {"units": "Pa"})
        out["density"] = ("property", [p.density for p in g.props], {"units": "kg m-3"})
        out["vp"] = ("property", [p.vp for p in g.props], {"units": "m/s"})
        out["vs"] = ("property", [p.vs for p in g.props], {"units": "m/s"})
    if g.sources:
        out["source_component"] = ("source", [s.component for s in g.sources])
        out["source_extent"] = (("source", "bound"), np.array([s.extent.as_tuple() for s in g.sources]))
    if g.stopes:
        out["stope_id"] = ("stope", [s.stope_id for s in g.stopes])
        out["stope_type"] = ("stope", [s.stype for s in g.stopes])
        out["stope_extent"] = (("stope", "bound"), np.array([s.extent.as_tuple() for s in g.stopes]))
    return out


def build_datatree(ds, include: list[str] | None = None) -> xr.DataTree:
    include = include or ["snapshots", "histories", "dumps", "geometry"]
    tree: dict[str, Any] = {"/": xr.Dataset(attrs={"source": str(ds.path), "name": ds.name,
                                                   "generator": "waveplot"})}
    if "snapshots" in include:
        for s in ds.series.values():
            tree[f"/snapshots/{group_name(s.slug)}"] = series_dataset(s)
    if "histories" in include and (h := histories_dataset(ds)) is not None:
        tree["/histories"] = h
    if "dumps" in include:
        for s in ds.dump_series.values():
            tree[f"/dumps/{group_name(s.slug)}"] = dump_dataset(s)
    if "geometry" in include and (g := geometry_dataset(ds)) is not None:
        tree["/geometry"] = g
    return xr.DataTree.from_dict(tree)


class WaveBackendEntrypoint(BackendEntrypoint):
    description = "WAVE simulation output (.map + .snp/.hst/.dmp/.geo)"
    url = "https://github.com/"
    open_dataset_parameters = ("filename_or_obj", "drop_variables")

    def guess_can_open(self, filename_or_obj) -> bool:
        return isinstance(filename_or_obj, str | bytes) or (hasattr(filename_or_obj, "__fspath__") and \
            str(filename_or_obj).lower().endswith(".map"))

    def open_dataset(self, filename_or_obj, *, drop_variables=None, group: str | None = None):  # type: ignore[override]
        tree = self.open_datatree(filename_or_obj)
        node = tree[group] if group else tree
        if group is None:
            # flat view: histories (if any) else the first snapshot series
            for name in ("histories",):
                if name in tree.children:
                    return tree[name].to_dataset()
            snaps = tree.children.get("snapshots")
            if snaps is not None and snaps.children:
                return next(iter(snaps.children.values())).to_dataset()
        return node.to_dataset()

    def open_datatree(self, filename_or_obj, **kwargs) -> xr.DataTree:  # type: ignore[override]
        from ..model.dataset import open_dataset

        return build_datatree(open_dataset(str(filename_or_obj)))
