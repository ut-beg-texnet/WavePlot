"""Reader for WAVE geometry files (``.geo``; ``DI_ReadGeom``, DATA_RSC.PAS:1367-1582).

The file is a sequence of 128-byte Fortran records ``[i4 120][120-byte payload][i4 120]``.
Starting at a geometry's byte offset: optional state / boundary blocks (first word
``0x01FEDCB1`` / ``0x01FEDCB2``), then ``proptot`` property blocks, ``mattot`` material
blocks, ``sourcetot`` source blocks and ``stopetot`` stope blocks (each stope optionally
followed by a byte layout padded to 128 bytes).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from .names import var_info
from .records import Extent, FormatIssue, GeomRecord

STATE_ID = 0x01FEDCB1
BOUND_ID = 0x01FEDCB2
BLOCK = 128


@dataclass(frozen=True)
class Prop:
    number: int
    bulk: float     # K (Pa)
    shear: float    # G (Pa)
    density: float  # rho (kg/m3)

    @property
    def vp(self) -> float:
        return math.sqrt(max(self.bulk + 4.0 * self.shear / 3.0, 0.0) / self.density) if self.density else 0.0

    @property
    def vs(self) -> float:
        return math.sqrt(max(self.shear, 0.0) / self.density) if self.density else 0.0

    @property
    def poisson(self) -> float:
        k, g = self.bulk, self.shear
        return (3 * k - 2 * g) / (2 * (3 * k + g)) if (3 * k + g) else 0.0


@dataclass(frozen=True)
class Material:
    number: int
    extent: Extent          # zone (cell) index ranges as written by WAVE, inclusive

    def bounds_m(self, axis: str, spacing: float) -> tuple[float, float]:
        """Physical edges along ``axis``: zones ``lo..hi`` span ``[(lo-1)·d, hi·d]``.

        Equivalent to v407's ``AdjustGeom`` (+1 on the upper index) followed by ``(X-1)·d``.
        """
        lo, hi = self.extent.range(axis)
        return (lo - 1) * spacing, hi * spacing


@dataclass(frozen=True)
class Source:
    stype: int              # WAVE variable number of the driven component
    extent: Extent
    params: dict[str, float] = field(default_factory=dict)

    @property
    def component(self) -> str:
        return var_info(self.stype).name if self.stype > 0 else f"type {self.stype}"


@dataclass
class Stope:
    stope_id: int
    stype: int
    extent: Extent
    params: dict[str, float] = field(default_factory=dict)
    layout: np.ndarray | None = None   # uint8 array shaped (nk, nj, ni); nonzero = open/mined (unverified)
    outline: list[tuple[int, int]] | None = None

    @property
    def plane(self) -> str | None:
        e = self.extent
        if e.i1 == e.i2:
            return "kj"
        if e.j1 == e.j2:
            return "ik"
        if e.k1 == e.k2:
            return "ij"
        return None


@dataclass
class Geometry:
    record: GeomRecord
    props: list[Prop] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    stopes: list[Stope] = field(default_factory=list)
    grid: Extent | None = None                  # from the state block
    physical_size: tuple[float, float, float] | None = None   # from the first boundary block (m)
    boundary_stress: dict[str, float] | None = None
    issues: list[FormatIssue] = field(default_factory=list)

    @property
    def model3d(self) -> bool:
        ext = self.grid or self.record.extent
        return ext.k2 != ext.k1

    @property
    def spacing(self) -> dict[str, float]:
        return self.record.spacing

    def prop_for(self, material_number: int) -> Prop | None:
        for p in self.props:
            if p.number == material_number:
                return p
        return None

    def source_centroid(self) -> tuple[int, int, int]:
        """Grid-index centroid of the sources (``AdjustGeom`` COG_i/j/k, integer division)."""
        n = len(self.sources)
        q = 2 * n if n else 2
        si = 1 + sum(s.extent.i1 for s in self.sources) + 1 + sum(s.extent.i2 for s in self.sources)
        sj = 1 + sum(s.extent.j1 for s in self.sources) + 1 + sum(s.extent.j2 for s in self.sources)
        sk = 1 + sum(s.extent.k1 for s in self.sources) + 1 + sum(s.extent.k2 for s in self.sources)
        return si // q, sj // q, sk // q

    def source_centroid_m(self) -> tuple[float, float, float]:
        """Physical centroid (m) of the source boxes, measured like grid coordinates (index-1)*d."""
        if not self.sources:
            return (math.nan, math.nan, math.nan)
        d = self.spacing
        c = []
        for a in "ijk":
            vals = [(s.extent.range(a)[0] + s.extent.range(a)[1]) / 2 for s in self.sources]
            c.append((sum(vals) / len(vals) - 1) * d[a])
        return tuple(c)  # type: ignore[return-value]


_SRC_PARAM_NAMES = {7: "begin", 8: "end", 12: "amplitude", 13: "period"}
_STOPE_PARAM_NAMES = {9: "kn", 10: "ks", 16: "friction_deg", 17: "cohesion", 18: "tan_friction",
                      20: "tensile_strength"}


def read_geometry(data: bytes, rec: GeomRecord, otl_outline: list[tuple[int, int]] | None = None) -> Geometry:
    """Parse the geometry described by ``rec`` from the full ``.geo`` file contents."""
    geom = Geometry(record=rec)
    start, end = rec.offset, rec.offset + rec.grlen
    if end > len(data):
        geom.issues.append(FormatIssue("geo", f"geometry {rec.number} extends past end of file", "error"))
        end = len(data)

    def payload(pos: int) -> bytes:
        return data[pos + 4:pos + 4 + 120] if pos + BLOCK <= len(data) else b""

    pos = start
    nboundary = 0
    while pos + BLOCK <= end:
        p = payload(pos)
        (ident,) = struct.unpack_from("<i", p, 0)
        if ident == STATE_ID:
            ints = struct.unpack_from("<12i", p, 4)
            geom.grid = Extent(*ints[6:12])
        elif ident == BOUND_ID:
            if nboundary == 0:
                vals = struct.unpack_from("<6i6f", p, 4)
                geom.physical_size = (vals[7], vals[9], vals[11])
            elif nboundary == 1:
                f = struct.unpack_from("<6f", p, 4)
                if any(f):
                    geom.boundary_stress = dict(zip(("s11", "s22", "s33", "s12", "s23", "s13"), f, strict=True))
            nboundary += 1
        else:
            break
        pos += BLOCK

    def blocks(count: int):
        nonlocal pos
        for _ in range(count):
            if pos + BLOCK > len(data):
                geom.issues.append(FormatIssue("geo", "geometry blocks truncated", "error"))
                return
            yield payload(pos)
            pos += BLOCK

    for n, p in enumerate(blocks(rec.proptot), start=1):
        bulk, shear, dens = struct.unpack_from("<3f", p, 0)
        geom.props.append(Prop(n, bulk, shear, dens))

    for p in blocks(rec.mattot):
        num, i1, i2, j1, j2, k1, k2 = struct.unpack_from("<7i", p, 0)
        geom.materials.append(Material(num, Extent(i1, i2, j1, j2, k1, k2)))

    for p in blocks(rec.sourcetot):
        stype, i1, i2, j1, j2, k1, k2 = struct.unpack_from("<7i", p, 0)
        floats = struct.unpack_from("<23f", p, 28)
        params = {name: floats[k - 7] for k, name in _SRC_PARAM_NAMES.items()}
        geom.sources.append(Source(stype, Extent(i1, i2, j1, j2, k1, k2), params))
    geom.sources.reverse()  # WAVE stores them in reverse definition order

    for _ in range(rec.stopetot):
        if pos + BLOCK > len(data):
            geom.issues.append(FormatIssue("geo", "stope blocks truncated", "error"))
            break
        p = payload(pos)
        sid, stype, i1, i2, j1, j2, k1, k2 = struct.unpack_from("<8i", p, 0)
        floats = struct.unpack_from("<30f", p, 0)
        params = {name: floats[k] for k, name in _STOPE_PARAM_NAMES.items()}
        (layout_sz,) = struct.unpack_from("<i", p, 116)
        pos += BLOCK
        stope = Stope(sid, stype, Extent(i1, i2, j1, j2, k1, k2), params, outline=otl_outline)
        if 0 < layout_sz <= len(data) - pos:
            # v407 reads the layout from the start of the next block's payload (DATA_RSC.PAS:1518-1526).
            raw = np.frombuffer(data, dtype=np.uint8, count=layout_sz, offset=pos + 4)
            e = stope.extent
            shape = (e.k2 - e.k1 + 1, e.j2 - e.j1 + 1, e.i2 - e.i1 + 1)
            if int(np.prod(shape)) == layout_sz:
                stope.layout = raw.reshape(shape).copy()
            else:
                geom.issues.append(FormatIssue("geo", f"stope {sid}: layout size {layout_sz} does not match "
                                               f"extent {shape}; layout ignored"))
            pos += layout_sz + (BLOCK - layout_sz % BLOCK) % BLOCK
        geom.stopes.append(stope)
    geom.stopes.reverse()
    return geom
