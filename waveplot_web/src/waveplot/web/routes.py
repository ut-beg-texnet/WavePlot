"""Small JSON API under ``/api/v1`` (dataset listing, metadata, binary-friendly frame access)."""

from __future__ import annotations

import base64
from urllib.parse import unquote

import numpy as np
from flask import abort, jsonify, request

from ..science.decimate import fit_grid
from . import state


def _ds_or_404(ds_id: str):
    ds = state.get_dataset(unquote(ds_id))
    if ds is None:
        abort(404, description="dataset not found")
    return ds


def register_routes(app) -> None:
    server = app.server

    @server.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok"})

    @server.get("/api/v1/datasets")
    def datasets():
        reg = state.registry()
        return jsonify([{"id": e.id, "root": e.root, "path": e.relative, "size": e.size, "mtime": e.mtime}
                        for e in reg.list_datasets(refresh=request.args.get("refresh") == "1")])

    @server.get("/api/v1/info/<path:ds_id>")
    def info(ds_id: str):
        ds = _ds_or_404(ds_id)
        out = ds.info()
        out.pop("folder", None)   # never expose absolute server paths
        out["series"] = [{"slug": s.slug, "label": s.label, "frames": len(s), "units": s.units}
                         for s in ds.series.values()]
        return jsonify(out)

    @server.get("/api/v1/frame/<path:ds_id>")
    def frame(ds_id: str):
        """``?series=<slug>&n=<frame>&w=&h=`` → ``{dtype, bdata, shape, time, zmin, zmax}``."""
        ds = _ds_or_404(ds_id)
        slug = request.args.get("series", "")
        s = ds.series.get(slug)
        if s is None:
            abort(404, description="series not found")
        try:
            n = int(request.args.get("n", 0))
        except ValueError:
            abort(400)
        if not 0 <= n < len(s):
            abort(404, description="frame out of range")
        a = s.frame(n)
        w = int(request.args.get("w", 4096))
        h = int(request.args.get("h", 4096))
        a = fit_grid(a, (max(w, 2), max(h, 2)), "mean")
        zt = np.ascontiguousarray(a.T.astype("<f4"))
        return jsonify({"dtype": "f4", "bdata": base64.b64encode(zt.tobytes()).decode("ascii"),
                        "shape": list(zt.shape), "time": float(s.records[n].time),
                        "zmin": float(np.nanmin(zt)), "zmax": float(np.nanmax(zt))})
