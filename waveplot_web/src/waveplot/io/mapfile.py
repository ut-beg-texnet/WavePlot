"""Reader for the WAVE ``.map`` index (``DI_ReadMap``, DATA_RSC.PAS:680-788).

Unlike the Pascal original there are no capacity limits (v407 silently overwrote the last slot
after 250 snapshots / 50 dumps / ...), every record marker is validated, and a truncated final
record (a WAVE run still in progress) is tolerated.
"""

from __future__ import annotations

import dataclasses
import os
import struct
from pathlib import Path

from .records import (
    MARKER,
    PL_CRACK,
    PL_DUMP,
    PL_GEOM,
    PL_HIST,
    PL_SNAP,
    REC_LEN,
    CrackRecord,
    DumpRecord,
    FormatIssue,
    GeomRecord,
    HistRecord,
    MapIndex,
    SnapRecord,
)

_HEAD = struct.Struct("<ii")
_TAIL = struct.Struct("<i")


def read_map(path: str | os.PathLike[str], func_base: int = 0xFFFF) -> MapIndex:
    path = Path(path)
    data = path.read_bytes()
    index = MapIndex(path=str(path))
    name = path.name

    nfull, rest = divmod(len(data), REC_LEN)
    if rest:
        index.issues.append(FormatIssue(name, f"{rest} trailing bytes ignored (truncated final record?)"))

    snap_off = hist_off = dump_off = geom_off = crack_off = 0
    for r in range(nfull):
        rec = data[r * REC_LEN:(r + 1) * REC_LEN]
        lead, pltype = _HEAD.unpack_from(rec, 0)
        (trail,) = _TAIL.unpack_from(rec, REC_LEN - 4)
        if lead != MARKER or trail != MARKER:
            index.issues.append(FormatIssue(name, f"record markers {lead}/{trail}, expected {MARKER}", record=r))
        body = rec[8:]
        if pltype == PL_SNAP:
            s = SnapRecord.unpack(body, r, len(index.snaps) + 1, func_base)
            s = dataclasses.replace(s, offset=snap_off)
            snap_off += s.nfloats
            index.snaps.append(s)
        elif pltype == PL_HIST:
            h = HistRecord.unpack(body, r, len(index.hists) + 1, func_base)
            h = dataclasses.replace(h, offset=hist_off)
            hist_off += h.nfloats
            index.hists.append(h)
        elif pltype == PL_DUMP:
            d = DumpRecord.unpack(body, r, len(index.dumps) + 1)
            d = dataclasses.replace(d, offset=dump_off)
            dump_off += d.nfloats
            index.dumps.append(d)
        elif pltype == PL_GEOM:
            g = GeomRecord.unpack(body, r, len(index.geoms) + 1)
            g = dataclasses.replace(g, offset=geom_off)
            geom_off += g.grlen
            index.geoms.append(g)
        elif pltype == PL_CRACK:
            c = CrackRecord.unpack(body, r, len(index.cracks) + 1)
            c = dataclasses.replace(c, offset=crack_off)
            crack_off += c.crlen
            index.cracks.append(c)
        else:
            index.issues.append(FormatIssue(name, f"unknown record type {pltype} skipped", record=r))

    index.nrecords = nfull
    index.expected_sizes = {
        "snp": 4 * snap_off,
        "hst": 4 * hist_off,
        "dmp": 4 * dump_off,
        "geo": geom_off,
        "crk": crack_off,
    }
    for rec in index.snaps:
        if rec.xqty <= 0 or rec.yqty <= 0:
            index.issues.append(FormatIssue(name, f"snapshot {rec.number} has empty shape", record=rec.index))
    for rec in index.dumps:
        if not rec.is_planar:
            index.issues.append(FormatIssue(
                name, f"dump {rec.number} is not planar ({rec.iqty}x{rec.jqty}x{rec.kqty}); "
                "only planar dumps are supported", record=rec.index))
    return index
