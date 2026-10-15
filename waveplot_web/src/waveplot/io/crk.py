"""Reader for WAVE crack dumps (``.crk``) — UNVERIFIED (no sample file available).

Layout per ``Ds_GraphCrackD`` (DSPGEOM.PAS:791-806): for each of ``stqty`` cracks a 128-byte
header block (at +4: ``stid, iqty, i1, i2, j1, j2, k1, k2``) followed by ``iqty`` element
records of ``vqty + 2`` float32 values (marker, ``vqty`` values, marker). Element fields follow
``CrackRec`` (DATA_RSC.PAS:250-258). Every read is bounds-checked.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .records import CrackRecord, Extent, FormatIssue

CRACK_FIELDS = ("vNs1", "vNs0", "vT1s1", "vT1s0", "uNs1", "uNs0", "uT1s1", "uT1s0",
                "sNs1", "sT1s1", "sT1s0", "sT1sh", "T1slp", "Taslp", "Nbnd", "Shbnd")


@dataclass
class CrackElements:
    stope_id: int
    extent: Extent
    values: np.ndarray            # (iqty, vqty) float32
    fields: tuple[str, ...]

    def field(self, name: str) -> np.ndarray:
        return self.values[:, self.fields.index(name)]

    def state(self) -> np.ndarray:
        """0 = stuck, 1 = slipping (Taslp > 0), 2 = open (normal stress == 0)."""
        out = np.zeros(len(self.values), dtype=np.int8)
        if "Taslp" in self.fields:
            out[self.field("Taslp") > 0] = 1
        if "sNs1" in self.fields:
            out[self.field("sNs1") == 0] = 2
        return out


def read_crack_dump(data: bytes, rec: CrackRecord) -> tuple[list[CrackElements], list[FormatIssue]]:
    issues: list[FormatIssue] = []
    out: list[CrackElements] = []
    pos = rec.offset
    fields = CRACK_FIELDS if rec.vqty == len(CRACK_FIELDS) else tuple(f"c{n + 1}" for n in range(rec.vqty))
    for _ in range(rec.stqty):
        if pos + 128 > len(data):
            issues.append(FormatIssue("crk", f"crack dump {rec.number}: header beyond end of file", "error"))
            break
        stid, iqty, i1, i2, j1, j2, k1, k2 = struct.unpack_from("<8i", data, pos + 4)
        pos += 128
        nbytes = iqty * (rec.vqty + 2) * 4
        if iqty < 0 or pos + nbytes > len(data):
            issues.append(FormatIssue("crk", f"crack dump {rec.number}: elements beyond end of file", "error"))
            break
        vals = np.frombuffer(data, dtype="<f4", count=iqty * (rec.vqty + 2), offset=pos)
        vals = vals.reshape(iqty, rec.vqty + 2)[:, 1:-1].copy()
        out.append(CrackElements(stid, Extent(i1, i2, j1, j2, k1, k2), vals, fields))
        pos += nbytes
    return out, issues
