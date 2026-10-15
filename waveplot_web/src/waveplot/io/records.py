"""Typed records of the WAVE ``.map`` index file.

Each ``.map`` record is 128 bytes: ``[i4 120][i4 PlType][116-byte body][i4 120]``. The body
starts at byte 8 and is unpacked with the struct formats below (byte-packed Turbo Pascal records,
little-endian, IEEE float32). Fields not present in the file (offsets, derived time bases) are
computed by :mod:`waveplot.io.mapfile`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .names import VarInfo, func_name, stagger_shift, var_info

REC_LEN = 128
MARKER = 120

PL_SNAP, PL_HIST, PL_DUMP, PL_GEOM, PL_CRACK = 0, 1, 2, 3, 4

SNAP_FMT = struct.Struct("<iff6i2c2f2i2i2f2i")   # 78 bytes
HIST_FMT = struct.Struct("<iff6icfii2f2i")        # 65 bytes
DUMP_FMT = struct.Struct("<f6i2c2f2i2ii2if2i")    # 78 bytes
GEOM_FMT = struct.Struct("<2i4ifi6i3fi")          # 72 bytes (Pascal reads 76; 4 spill bytes unused)
CRACK_FMT = struct.Struct("<f2i2i3ii")            # 36 bytes

AXES = ("i", "j", "k")


def _ch(b: bytes) -> str:
    return b.decode("latin-1").lower() if b not in (b"\x00", b"") else "?"


@dataclass(frozen=True)
class Extent:
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int

    def range(self, axis: str) -> tuple[int, int]:
        return {"i": (self.i1, self.i2), "j": (self.j1, self.j2), "k": (self.k1, self.k2)}[axis]

    def contains(self, other: Extent, tol: int = 0) -> bool:
        return all(
            self.range(a)[0] - tol <= other.range(a)[0] and other.range(a)[1] <= self.range(a)[1] + tol
            for a in AXES
        )

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.i1, self.i2, self.j1, self.j2, self.k1, self.k2)


@dataclass(frozen=True)
class SnapRecord:
    index: int          # position in the .map file (0-based)
    number: int         # 1-based number among snapshots (legacy numbering)
    var_raw: int
    var: int
    func: int
    t1: float
    time: float
    extent: Extent
    ax1: str
    ax2: str
    dx: float
    dy: float
    sx: int
    sy: int
    xqty: int
    yqty: int
    maxval: float
    minval: float
    gnum: int
    gid: int
    offset: int = 0     # in float32 units into the .snp file

    @classmethod
    def unpack(cls, body: bytes, index: int, number: int, func_base: int) -> SnapRecord:
        (raw, t1, time, i1, i2, j1, j2, k1, k2, a1, a2, dx, dy, sx, sy, xq, yq, mx, mn, gnum,
         gid) = SNAP_FMT.unpack_from(body)
        var, func = _split(raw, func_base)
        return cls(index, number, raw, var, func, t1, time, Extent(i1, i2, j1, j2, k1, k2),
                   _ch(a1), _ch(a2), dx, dy, sx, sy, xq, yq, mx, mn, gnum, gid)

    @property
    def info(self) -> VarInfo:
        return var_info(self.var)

    @property
    def name(self) -> str:
        f = func_name(self.func)
        return self.info.name + (f" {f}" if f else "")

    @property
    def units(self) -> str:
        return self.info.units

    @property
    def axes(self) -> str:
        return self.ax1 + self.ax2

    @property
    def is_spacetime(self) -> bool:
        return "t" in (self.ax1, self.ax2)

    @property
    def fixed_axis(self) -> str | None:
        rest = [a for a in AXES if a not in (self.ax1, self.ax2)]
        return rest[0] if len(rest) == 1 else None

    @property
    def fixed_index(self) -> int | None:
        a = self.fixed_axis
        return self.extent.range(a)[0] if a else None

    @property
    def stagger(self) -> tuple[int, int]:
        return stagger_shift(self.info.position, self.ax1, self.ax2)

    @property
    def nfloats(self) -> int:
        return self.xqty * (self.yqty + 2)


@dataclass(frozen=True)
class HistRecord:
    index: int
    number: int
    var_raw: int
    var: int
    func: int
    t1: float
    t2: float
    extent: Extent
    ax1: str
    dx: float
    sx: int
    xqty: int
    maxval: float
    minval: float
    gnum: int
    gid: int
    offset: int = 0     # float32 units into the .hst file

    @classmethod
    def unpack(cls, body: bytes, index: int, number: int, func_base: int) -> HistRecord:
        (raw, t1, t2, i1, i2, j1, j2, k1, k2, a1, dx, sx, xq, mx, mn, gnum, gid) = HIST_FMT.unpack_from(body)
        var, func = _split(raw, func_base)
        return cls(index, number, raw, var, func, t1, t2, Extent(i1, i2, j1, j2, k1, k2), _ch(a1), dx, sx,
                   xq, mx, mn, gnum, gid)

    @property
    def info(self) -> VarInfo:
        return var_info(self.var)

    @property
    def name(self) -> str:
        f = func_name(self.func)
        return self.info.name + (f" {f}" if f else "")

    @property
    def units(self) -> str:
        return self.info.units

    @property
    def is_profile(self) -> bool:
        """Spatial line profile along i/j/k instead of a time series."""
        return self.ax1 in AXES

    @property
    def tstart(self) -> float:
        if self.is_profile:
            return (self.extent.range(self.ax1)[0] - 1) * self.dx
        return self.t1

    @property
    def tend(self) -> float:
        if self.is_profile:
            return (self.extent.range(self.ax1)[1] - 1) * self.dx
        return self.t2

    @property
    def n(self) -> int:
        return self.xqty

    @property
    def dt(self) -> float:
        return (self.tend - self.tstart) / (self.n - 1) if self.n > 1 else 0.0

    @property
    def x_unit(self) -> str:
        return "m" if self.is_profile else "s"

    @property
    def position(self) -> tuple[int, int, int]:
        return (self.extent.i1, self.extent.j1, self.extent.k1)

    @property
    def is_global(self) -> bool:
        return self.position == (0, 0, 0)

    @property
    def vmax(self) -> float:
        return max(abs(self.maxval), abs(self.minval))

    @property
    def nfloats(self) -> int:
        return 3 * self.xqty


# In-plane component selection for dumps (DATA_RSC.PAS:1147-1151): plane -> (v1, v2, s11, s22, s12)
DUMP_PLANE_COMPONENTS: dict[str, tuple[str, str, str, str, str]] = {
    "ij": ("v1", "v2", "s11", "s22", "s12"),
    "ik": ("v1", "v3", "s11", "s33", "s13"),
    "kj": ("v3", "v2", "s33", "s22", "s23"),
}
DUMP_COMPONENTS_3D = ("v1", "v2", "v3", "s11", "s22", "s33", "s12", "s23", "s13")
DUMP_COMPONENTS_2D = ("v1", "v2", "s11", "s22", "s12")


@dataclass(frozen=True)
class DumpRecord:
    index: int
    number: int
    time: float
    extent: Extent
    ax1_file: str
    ax2_file: str
    dx: float
    dy: float
    sx: int
    sy: int
    iqty: int
    jqty: int
    vqty: int
    gnum: int
    gid: int
    dz: float
    sz: int
    kqty: int
    offset: int = 0     # float32 units into the .dmp file (data starts at offset + 1)

    @classmethod
    def unpack(cls, body: bytes, index: int, number: int) -> DumpRecord:
        (time, i1, i2, j1, j2, k1, k2, a1, a2, dx, dy, sx, sy, iq, jq, vq, gnum, gid, dz, sz,
         kq) = DUMP_FMT.unpack_from(body)
        if kq == 0:  # backwards compatibility (DATA_RSC.PAS:745)
            kq, sz, dz = 1, 1, dx
        return cls(index, number, time, Extent(i1, i2, j1, j2, k1, k2), _ch(a1), _ch(a2), dx, dy, sx, sy,
                   iq, jq, vq, gnum, gid, dz, sz, kq)

    @property
    def dim3(self) -> bool:
        return self.vqty == 9

    @property
    def components(self) -> tuple[str, ...]:
        if self.vqty == 9:
            return DUMP_COMPONENTS_3D
        if self.vqty == 5:
            return DUMP_COMPONENTS_2D
        return tuple(f"c{n + 1}" for n in range(self.vqty))

    # Plane axes follow Ds_DumpScale (DSPL.PAS:843-848), not the file's Ax1/Ax2.
    @property
    def ax1(self) -> str:
        return "k" if (self.kqty > 1 and self.jqty > 1) else "i"

    @property
    def ax2(self) -> str:
        return "j" if self.jqty > 1 else "k"

    @property
    def axes(self) -> str:
        return self.ax1 + self.ax2

    @property
    def xqty(self) -> int:
        return {"i": self.iqty, "j": self.jqty, "k": self.kqty}[self.ax1]

    @property
    def yqty(self) -> int:
        return {"i": self.iqty, "j": self.jqty, "k": self.kqty}[self.ax2]

    @property
    def steps(self) -> tuple[int, int]:
        s = {"i": self.sx, "j": self.sy, "k": self.sz}
        return s[self.ax1], s[self.ax2]

    @property
    def spacing(self) -> tuple[float, float]:
        d = {"i": self.dx, "j": self.dy, "k": self.dz}
        return d[self.ax1], d[self.ax2]

    @property
    def fixed_axis(self) -> str | None:
        rest = [a for a in AXES if a not in (self.ax1, self.ax2)]
        return rest[0] if len(rest) == 1 else None

    @property
    def fixed_index(self) -> int | None:
        a = self.fixed_axis
        return self.extent.range(a)[0] if a else None

    @property
    def is_planar(self) -> bool:
        return self.iqty * self.jqty * self.kqty == self.xqty * self.yqty

    @property
    def nfloats(self) -> int:
        return 2 + self.iqty * self.jqty * self.kqty * self.vqty


@dataclass(frozen=True)
class GeomRecord:
    index: int
    number: int
    gnum: int
    gid: int
    proptot: int
    mattot: int
    sourcetot: int
    stopetot: int
    time: float
    ntot: int
    extent: Extent
    dx: float
    dy: float
    dz: float
    grlen: int
    offset: int = 0     # bytes into the .geo file

    @classmethod
    def unpack(cls, body: bytes, index: int, number: int) -> GeomRecord:
        (gnum, gid, prop, mat, src, stope, time, ntot, i1, i2, j1, j2, k1, k2, dx, dy, dz,
         grlen) = GEOM_FMT.unpack_from(body)
        if grlen == 0:
            grlen = (prop + mat + src + stope) * REC_LEN
        if dx == 0:
            dx = dy = dz = 1.0
        return cls(index, number, gnum, gid, prop, mat, src, stope, time, ntot,
                   Extent(i1, i2, j1, j2, k1, k2), dx, dy, dz, grlen)

    @property
    def model3d(self) -> bool:
        return self.extent.k2 != self.extent.k1

    @property
    def spacing(self) -> dict[str, float]:
        return {"i": self.dx, "j": self.dy, "k": self.dz}


@dataclass(frozen=True)
class CrackRecord:
    index: int
    number: int
    time: float
    ic1: int
    ic2: int
    gnum: int
    gid: int
    stqty: int
    tqty: int
    vqty: int
    crlen: int
    offset: int = 0     # bytes into the .crk file

    @classmethod
    def unpack(cls, body: bytes, index: int, number: int) -> CrackRecord:
        (time, ic1, ic2, gnum, gid, stq, tq, vq, crlen) = CRACK_FMT.unpack_from(body)
        return cls(index, number, time, ic1, ic2, gnum, gid, stq, tq, vq, crlen)


@dataclass
class FormatIssue:
    file: str
    message: str
    severity: str = "warning"   # "warning" | "error"
    record: int | None = None

    def __str__(self) -> str:
        where = f" (record {self.record})" if self.record is not None else ""
        return f"[{self.severity}] {self.file}{where}: {self.message}"


@dataclass
class MapIndex:
    path: str
    snaps: list[SnapRecord] = field(default_factory=list)
    hists: list[HistRecord] = field(default_factory=list)
    dumps: list[DumpRecord] = field(default_factory=list)
    geoms: list[GeomRecord] = field(default_factory=list)
    cracks: list[CrackRecord] = field(default_factory=list)
    issues: list[FormatIssue] = field(default_factory=list)
    nrecords: int = 0
    # Expected data-file sizes in bytes implied by the accumulated offsets
    expected_sizes: dict[str, int] = field(default_factory=dict)


def _split(raw: int, base: int) -> tuple[int, int]:
    if raw < 0:
        return raw, 0
    return raw % base, raw // base
