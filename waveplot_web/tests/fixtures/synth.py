"""Spec-conformant writer for tiny synthetic WAVE datasets (used where no sample files exist:
2-D dumps, time-axis snapshots, function codes, crack dumps, upper-case extensions, ...)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from waveplot.io.records import CRACK_FMT, DUMP_FMT, GEOM_FMT, HIST_FMT, SNAP_FMT

FUNC_BASE = 0xFFFF


def _rec(pltype: int, body: bytes) -> bytes:
    body = body.ljust(116, b"z")
    assert len(body) == 116
    return struct.pack("<ii", 120, pltype) + body + struct.pack("<i", 120)


def _fortran(payload: bytes) -> bytes:
    n = len(payload)
    return struct.pack("<i", n) + payload + struct.pack("<i", n)


@dataclass
class SynthDataset:
    map_records: list[bytes] = field(default_factory=list)
    snp: bytearray = field(default_factory=bytearray)
    hst: bytearray = field(default_factory=bytearray)
    dmp: bytearray = field(default_factory=bytearray)
    geo: bytearray = field(default_factory=bytearray)
    crk: bytearray = field(default_factory=bytearray)

    def snapshot(self, data: np.ndarray, var: int, time: float, axes: str = "ij",
                 extent=(1, 1, 1, 1, 1, 1), dx: float = 1.0, dy: float = 1.0, sx: int = 1, sy: int = 1,
                 func: int = 0, t1: float | None = None, gnum: int = 1, gid: int = 7) -> None:
        data = np.asarray(data, dtype="<f4")
        xq, yq = data.shape
        raw = var + func * FUNC_BASE
        body = SNAP_FMT.pack(raw, time if t1 is None else t1, time, *extent, axes[0].encode(), axes[1].encode(),
                             dx, dy, sx, sy, xq, yq, float(data.max()), float(data.min()), gnum, gid)
        self.map_records.append(_rec(0, body))
        for row in data:
            self.snp += _fortran(row.tobytes())

    def history(self, values: np.ndarray, var: int, t1: float, t2: float, pos=(1, 1, 1), ax: str = "t",
                extent=None, dx: float = 0.01, sx: int = 1, func: int = 0) -> None:
        values = np.asarray(values, dtype="<f4")
        ext = extent or (pos[0], pos[0], pos[1], pos[1], pos[2], pos[2])
        body = HIST_FMT.pack(var + func * FUNC_BASE, t1, t2, *ext, ax.encode(), dx, sx, len(values),
                             float(values.max()), float(values.min()), 1, 7)
        self.map_records.append(_rec(1, body))
        for v in values:
            self.hst += _fortran(v.tobytes())

    def dump(self, data: np.ndarray, time: float, extent, iqty: int, jqty: int, kqty: int,
             dx: float = 1.0, dy: float = 1.0, dz: float = 1.0, write_kqty: bool = True) -> None:
        data = np.asarray(data, dtype="<f4")
        vqty = data.shape[-1]
        body = DUMP_FMT.pack(time, *extent, b"i", b"j", dx, dy, 1, 1, iqty, jqty, vqty, 1, 7, dz, 1,
                             kqty if write_kqty else 0)
        self.map_records.append(_rec(2, body))
        self.dmp += _fortran(data.tobytes())

    def geometry(self, props, mats, sources, stopes, extent, d=(1.0, 1.0, 1.0), grid=None) -> None:
        blocks = bytearray()

        def block(payload: bytes) -> None:
            nonlocal blocks
            blocks += _fortran(payload.ljust(120, b"\0"))

        g = grid or extent
        block(struct.pack("<i", 0x01FEDCB1) + struct.pack("<12i", 0, 0, 1, 0, 0, 0, *g))
        block(struct.pack("<i", 0x01FEDCB2) + struct.pack("<6i6f", *g, 0, 10.0, 0, 20.0, 0, 30.0))
        for k, g_, rho in props:
            block(struct.pack("<3f", k, g_, rho))
        for m in mats:
            block(struct.pack("<7i", *m))
        for s in reversed(sources):
            block(struct.pack("<7i", *s).ljust(28, b"\0") + struct.pack("<9f", -1, 1, 0, 0, 0, 1e-10, 0.1, 0, 1))
        for sid, stype, ext, layout in reversed(stopes):
            payload = bytearray(struct.pack("<8i", sid, stype, *ext).ljust(120, b"\0"))
            n = 0 if layout is None else layout.size
            struct.pack_into("<i", payload, 116, n)
            block(bytes(payload))
            if layout is not None:
                # v407 reads the layout from the next block start + 4 and skips n + pad bytes
                pad = (128 - n % 128) % 128
                assert pad >= 4, "choose a layout size leaving room for the record marker"
                region = struct.pack("<i", n) + layout.astype(np.uint8).tobytes()
                blocks += region.ljust(n + pad, b"\0")
        body = GEOM_FMT.pack(1, 7, len(props), len(mats), len(sources), len(stopes), 0.0, 0, *extent, *d,
                             len(blocks))
        self.map_records.append(_rec(3, body))
        self.geo += blocks

    def crack(self, time: float, cracks: list[tuple[int, tuple, np.ndarray]]) -> None:
        start = len(self.crk)
        for stid, ext, vals in cracks:
            iqty, _vqty = vals.shape
            self.crk += _fortran(struct.pack("<8i", stid, iqty, *ext).ljust(120, b"\0"))
            for row in vals.astype("<f4"):
                self.crk += _fortran(row.tobytes())
        crlen = len(self.crk) - start
        body = CRACK_FMT.pack(time, 1, 1, 1, 7, len(cracks), sum(v.shape[0] for _, _, v in cracks),
                              cracks[0][2].shape[1] if cracks else 16, crlen)
        self.map_records.append(_rec(4, body))

    def write(self, folder: Path, stem: str = "synth", upper: bool = False) -> Path:
        folder.mkdir(parents=True, exist_ok=True)

        def name(ext: str) -> Path:
            return folder / f"{stem}.{ext.upper() if upper else ext}"

        name("map").write_bytes(b"".join(self.map_records))
        for ext, buf in (("snp", self.snp), ("hst", self.hst), ("dmp", self.dmp), ("geo", self.geo),
                         ("crk", self.crk)):
            if buf:
                name(ext).write_bytes(bytes(buf))
        return name("map")
