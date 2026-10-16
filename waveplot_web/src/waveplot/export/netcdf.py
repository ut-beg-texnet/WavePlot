"""Conversion of a whole dataset to netCDF4/HDF5 (``waveplot convert --to netcdf``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def to_netcdf(ds, target: str | Path, include: list[str] | None = None,
              progress: Callable[[str], None] | None = None) -> Path:
    from ..io.xr_backend import build_datatree

    tree = build_datatree(ds, include)
    target = Path(target)
    if progress:
        progress(f"writing {sum(1 for _ in tree.subtree)} groups to {target.name} …")
    encoding = {}
    for node in tree.subtree:
        enc = {}
        for name, var in node.data_vars.items():
            if var.dtype.kind == "f" and var.ndim >= 2:
                enc[name] = {"zlib": True, "complevel": 4, "shuffle": True}
        if enc:
            encoding[node.path] = enc
    tree.to_netcdf(target, engine="h5netcdf", encoding=encoding)
    return target
