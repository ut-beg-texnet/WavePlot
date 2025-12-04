"""
Full .MAP parser for WavePlot data structures.

This module mirrors the Turbo Pascal structures and read logic defined in
`Turbo pascal version/DATA_RSC.PAS` (see SnapMapRec, HistMapRec, DumpMapRec,
GeomMapRec, CrackDMapRec) and reconstructs the in-memory maps of records with
derived fields such as offsets and function codes.

Notes on file layout (per DATA_RSC.PAS DI_ReadMap):
- Each record in the .MAP file begins with an 8-byte header:
  1) ignored 4-byte little-endian integer (dummy)
  2) 4-byte little-endian integer `PlType` indicating record type:
     0 = Snapshot, 1 = History, 2 = Dump, 3 = Geometry, 4 = Crack data
- This is followed by a fixed 120-byte record body. Only the leading portion
  of each body contains the fields defined below; the remaining bytes are
  reserved/padding and are skipped by the Pascal code as well.

Offsets into the companion binary files are reconstructed exactly like in
Pascal:
- Snapshots (.SNP): Offset accumulates by Xqty * (Yqty + 2)
- Histories (.HST): Offset accumulates by Xqty * 3
- Dumps (.DMP): Offset accumulates by 1 + iqty*jqty*kqty * Vqty + 1
- Geometries (.GEO): Offset accumulates by GRlen (bytes)
- Crack data (.CRK): Offset accumulates by CrLen (bytes)

Caveats:
- Some fields (e.g., QuickX/QuickY) live outside the 78/65/76/36 byte segments
  that Pascal reads, so they are unavailable in the .MAP body and are left as
  None here. Function codes (SFunc, HFunc) are derived from the packed Var
  numbers as done in DATA_RSC.PAS (DIV/MOD 0xFFFF).

"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import List, Optional


# Fixed size of the body per record in .MAP files (padding included)
MAP_RECORD_BODY_BYTES = 120


@dataclass
class SnapMapRecord:
    # Computed/housekeeping
    offset: int  # offset (in float32s) within .SNP data stream
    # Values passed from WAVE (from .MAP)
    snap_var: int
    sfunc: int
    t1: float
    time: float
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    ax1: str
    ax2: str
    dx: float
    dy: float
    sx: int
    sy: int
    xqty: int
    yqty: int
    max_val: float
    min_val: float
    gnum: int
    gid: int
    # Optional (not stored in the 78-byte segment)
    quick_x: Optional[int] = None
    quick_y: Optional[int] = None


@dataclass
class HistMapRecord:
    # Computed/housekeeping
    offset: int  # offset (in float32s) within .HST data stream
    # Values passed from WAVE (from .MAP)
    hist_var: int
    hfunc: int
    t1: float
    t2: float
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    ax1: str
    dx: float
    sx: int
    xqty: int
    max_val: float
    min_val: float
    gnum: int
    gid: int
    # Derived as in DATA_RSC.PAS
    tqty: int
    xp: int
    yp: int
    zp: int
    tstart: float
    time: float
    dt: float
    vmax: float


@dataclass
class DumpMapRecord:
    # Computed/housekeeping
    offset: int  # offset (in float32s) within .DMP data stream
    # Values passed from WAVE (from .MAP)
    time: float
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    ax1: str
    ax2: str
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
    dim3: bool
    # Additional/derived
    max_val: Optional[float]
    xqty: int
    yqty: int
    quick_x: Optional[int] = None
    quick_y: Optional[int] = None


@dataclass
class GeomMapRecord:
    # Computed/housekeeping
    offset: int  # byte offset within .GEO data stream
    # Values passed from WAVE (from .MAP)
    gnum: int
    gid: int
    proptot_t: int
    mattot_t: int
    sourcetot_t: int
    stopetot_t: int
    time: float
    ntot: int
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    dx: float
    dy: float
    dz: float
    grlen: int
    # Derived like in DATA_RSC.PAS
    model3d: bool


@dataclass
class CrackDMapRecord:
    # Computed/housekeeping
    offset: int  # byte offset within .CRK data stream
    # Values passed from WAVE (from .MAP)
    time: float
    ic1: int
    ic2: int
    gnum: int
    gid: int
    stqty: int
    tqty: int
    vqty: int
    crlen: int


@dataclass
class MapFileParsed:
    snaps: List[SnapMapRecord]
    hists: List[HistMapRecord]
    dumps: List[DumpMapRecord]
    geoms: List[GeomMapRecord]
    cracks: List[CrackDMapRecord]

    @property
    def counts(self):
        return {
            "snapshots": len(self.snaps),
            "histories": len(self.hists),
            "dumps": len(self.dumps),
            "geometries": len(self.geoms),
            "crack_data": len(self.cracks),
        }


def _char_from_byte(b: bytes) -> str:
    try:
        return b.decode('latin-1') if b else ''
    except Exception:
        return ''


def parse_map_file(map_path: str) -> MapFileParsed:
    """
    Parse a .MAP file fully into record lists mirroring DATA_RSC.PAS structures.

    Returns MapFileParsed with arrays of the various record types and computed
    offsets suitable for reading the companion .SNP/.HST/.DMP/.GEO/.CRK files.
    """
    if not os.path.exists(map_path):
        raise FileNotFoundError(f".map file not found: {map_path}")

    snaps: List[SnapMapRecord] = []
    hists: List[HistMapRecord] = []
    dumps: List[DumpMapRecord] = []
    geoms: List[GeomMapRecord] = []
    cracks: List[CrackDMapRecord] = []

    # Accumulated offsets (match Pascal logic)
    snap_start = 0  # in float32 elements
    hist_start = 0  # in float32 elements
    dump_start = 0  # in float32 elements
    geom_start = 0  # in bytes
    crack_start = 0 # in bytes

    with open(map_path, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            try:
                _dum, pl_type = struct.unpack('<II', header)
            except struct.error:
                break

            body = f.read(MAP_RECORD_BODY_BYTES)
            if len(body) < MAP_RECORD_BODY_BYTES:
                break

            # Snapshot record
            if pl_type == 0:
                # First 78 bytes contain all useful fields
                head = body[:78]
                (
                    snapvar_raw,
                    t1, time_val,
                    i1, i2, j1, j2, k1, k2,
                    ax1_b, ax2_b,
                    dx, dy,
                    sx, sy,
                    xqty, yqty,
                    max_val, min_val,
                    gnum, gid,
                ) = struct.unpack('<lff' + 'l'*6 + 'cc' + 'ff' + 'l'*4 + 'ff' + 'll', head)

                sfunc = int(snapvar_raw // 0xFFFF)
                snap_var = int(snapvar_raw % 0xFFFF)
                ax1 = _char_from_byte(ax1_b)
                ax2 = _char_from_byte(ax2_b)

                rec = SnapMapRecord(
                    offset=snap_start,
                    snap_var=snap_var,
                    sfunc=sfunc,
                    t1=float(t1),
                    time=float(time_val),
                    i1=int(i1), i2=int(i2), j1=int(j1), j2=int(j2), k1=int(k1), k2=int(k2),
                    ax1=ax1, ax2=ax2,
                    dx=float(dx), dy=float(dy),
                    sx=int(sx), sy=int(sy),
                    xqty=int(xqty), yqty=int(yqty),
                    max_val=float(max_val), min_val=float(min_val),
                    gnum=int(gnum), gid=int(gid),
                )
                snaps.append(rec)
                # Advance snapshot offset (Pascal: + Xqty * (Yqty + 2))
                try:
                    snap_start += int(xqty) * (int(yqty) + 2)
                except Exception:
                    pass

            # History record
            elif pl_type == 1:
                head = body[:65]
                (
                    histvar_raw,
                    t1, t2,
                    i1, i2, j1, j2, k1, k2,
                    ax1_b,
                    dx,
                    sx,
                    xqty,
                    max_val, min_val,
                    gnum, gid,
                ) = struct.unpack('<lff' + 'l'*6 + 'c' + 'f' + 'l' + 'l' + 'ff' + 'll', head)

                hfunc = int(histvar_raw // 0xFFFF)
                hist_var = int(histvar_raw % 0xFFFF)
                ax1 = _char_from_byte(ax1_b)

                # Derived values (see DATA_RSC.PAS)
                tqty = int(xqty)
                xp, yp, zp = int(i1), int(j1), int(k1)
                if ax1 in ('i', 'I'):
                    tstart = (int(i1) - 1) * float(dx)
                    time_val = (int(i2) - 1) * float(dx)
                elif ax1 in ('j', 'J'):
                    tstart = (int(j1) - 1) * float(dx)
                    time_val = (int(j2) - 1) * float(dx)
                elif ax1 in ('k', 'K'):
                    tstart = (int(k1) - 1) * float(dx)
                    time_val = (int(k2) - 1) * float(dx)
                else:
                    tstart = float(t1)
                    time_val = float(t2)
                dt = 0.0
                try:
                    if tqty and (tqty > 1):
                        dt = (float(time_val) - float(tstart)) / (tqty - 1)
                except Exception:
                    dt = 0.0
                vmax = max(abs(float(max_val)), abs(float(min_val)))

                rec = HistMapRecord(
                    offset=hist_start,
                    hist_var=hist_var,
                    hfunc=hfunc,
                    t1=float(t1), t2=float(t2),
                    i1=int(i1), i2=int(i2), j1=int(j1), j2=int(j2), k1=int(k1), k2=int(k2),
                    ax1=ax1,
                    dx=float(dx), sx=int(sx), xqty=int(xqty),
                    max_val=float(max_val), min_val=float(min_val),
                    gnum=int(gnum), gid=int(gid),
                    tqty=tqty,
                    xp=xp, yp=yp, zp=zp,
                    tstart=float(tstart), time=float(time_val), dt=float(dt),
                    vmax=float(vmax),
                )
                hists.append(rec)
                # Advance history offset (Pascal: + Xqty * 3)
                try:
                    hist_start += int(xqty) * 3
                except Exception:
                    pass

            # Dump record
            elif pl_type == 2:
                head = body[:78]
                (
                    time_val,
                    i1, i2, j1, j2, k1, k2,
                    ax1_b, ax2_b,
                    dx, dy,
                    sx, sy,
                    iqty, jqty,
                    vqty,
                    gnum, gid,
                    dz,
                    sz, kqty,
                ) = struct.unpack('<f' + 'l'*6 + 'cc' + 'ff' + 'll' + 'l' + 'll' + 'f' + 'll', head)

                ax1 = _char_from_byte(ax1_b)
                ax2 = _char_from_byte(ax2_b)
                dim3 = bool(int(vqty) == 9)

                # Compatibility tweaks as in Pascal
                if int(kqty) == 0:
                    kqty = 1
                    sz = 1
                    dz = float(dx)

                rec = DumpMapRecord(
                    offset=dump_start,
                    time=float(time_val),
                    i1=int(i1), i2=int(i2), j1=int(j1), j2=int(j2), k1=int(k1), k2=int(k2),
                    ax1=ax1, ax2=ax2,
                    dx=float(dx), dy=float(dy),
                    sx=int(sx), sy=int(sy),
                    iqty=int(iqty), jqty=int(jqty),
                    vqty=int(vqty),
                    gnum=int(gnum), gid=int(gid),
                    dz=float(dz),
                    sz=int(sz), kqty=int(kqty),
                    dim3=dim3,
                    max_val=None,
                    xqty=int(iqty), yqty=int(jqty),
                )
                dumps.append(rec)
                # Advance dump offset (Pascal: + 1 + iqty*jqty*kqty * Vqty + 1)
                try:
                    dump_start += 1 + int(iqty) * int(jqty) * int(rec.kqty) * int(vqty) + 1
                except Exception:
                    pass

            # Geometry record
            elif pl_type == 3:
                head = body[:76]
                (
                    gnum, gid,
                    proptot_t, mattot_t, sourcetot_t, stopetot_t,
                    time_val, ntot,
                    i1, i2, j1, j2, k1, k2,
                    dx, dy, dz,
                    grlen,
                ) = struct.unpack('<' + 'l'*2 + 'l'*4 + 'f' + 'l' + 'l'*6 + 'f'*3 + 'l', head)

                # Back-compat: if grlen==0, derive as in Pascal
                if int(grlen) == 0:
                    grlen = int((int(proptot_t) + int(mattot_t) + int(sourcetot_t) + int(stopetot_t)) * 128)
                # dx/dy/dz must be non-zero (Pascal back-compat)
                if float(dx) == 0.0:
                    dx = 1.0; dy = 1.0; dz = 1.0

                model3d = not (int(k2) == int(k1))

                rec = GeomMapRecord(
                    offset=geom_start,
                    gnum=int(gnum), gid=int(gid),
                    proptot_t=int(proptot_t), mattot_t=int(mattot_t), sourcetot_t=int(sourcetot_t), stopetot_t=int(stopetot_t),
                    time=float(time_val), ntot=int(ntot),
                    i1=int(i1), i2=int(i2), j1=int(j1), j2=int(j2), k1=int(k1), k2=int(k2),
                    dx=float(dx), dy=float(dy), dz=float(dz),
                    grlen=int(grlen),
                    model3d=bool(model3d),
                )
                geoms.append(rec)
                try:
                    geom_start += int(grlen)
                except Exception:
                    pass

            # Crack data record
            elif pl_type == 4:
                head = body[:36]
                (
                    time_val,
                    ic1, ic2,
                    gnum, gid,
                    stqty, tqty, vqty,
                    crlen,
                ) = struct.unpack('<f' + 'l'*2 + 'l'*2 + 'l'*3 + 'l', head)

                rec = CrackDMapRecord(
                    offset=crack_start,
                    time=float(time_val),
                    ic1=int(ic1), ic2=int(ic2),
                    gnum=int(gnum), gid=int(gid),
                    stqty=int(stqty), tqty=int(tqty), vqty=int(vqty),
                    crlen=int(crlen),
                )
                cracks.append(rec)
                try:
                    crack_start += int(crlen)
                except Exception:
                    pass

            # Unknown/unsupported type: ignore
            else:
                continue

    return MapFileParsed(snaps=snaps, hists=hists, dumps=dumps, geoms=geoms, cracks=cracks)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Parse a .MAP file fully and print a summary (and optional JSON dump).')
    parser.add_argument('map_file', type=str, help='Path to the .map file')
    parser.add_argument('--json', dest='json_out', type=str, default=None, help='Optional path to write parsed content as JSON')
    args = parser.parse_args()

    parsed = parse_map_file(args.map_file)
    print('MAP counts:')
    for k, v in parsed.counts.items():
        print(f'  {k:>11}: {v}')

    # Show a couple of examples for quick sanity
    if parsed.snaps:
        s = parsed.snaps[0]
        print(f"\nFirst SNAP: snap_var={s.snap_var} sfunc={s.sfunc} axes={s.ax1}{s.ax2} grid={s.xqty}x{s.yqty} time={s.time}")
    if parsed.hists:
        h = parsed.hists[0]
        print(f"First HIST: hist_var={h.hist_var} hfunc={h.hfunc} axis={h.ax1} tqty={h.tqty} dt={h.dt}")
    if parsed.dumps:
        d = parsed.dumps[0]
        print(f"First DUMP: vqty={d.vqty} dim3={d.dim3} plane={d.ax1}{d.ax2} size={d.iqty}x{d.jqty}x{d.kqty}")
    if parsed.geoms:
        g = parsed.geoms[0]
        print(f"First GEOM: gnum={g.gnum} gid={g.gid} dims=({g.i1},{g.i2})x({g.j1},{g.j2})x({g.k1},{g.k2}) grlen={g.grlen}")
    if parsed.cracks:
        c = parsed.cracks[0]
        print(f"First CRACK: t={c.time} stqty={c.stqty} tqty={c.tqty} vqty={c.vqty} crlen={c.crlen}")

    if args.json_out:
        # Convert dataclasses to serializable dicts
        def dc_list(objs):
            from dataclasses import asdict
            return [asdict(o) for o in objs]

        out = {
            'counts': parsed.counts,
            'snaps': dc_list(parsed.snaps),
            'hists': dc_list(parsed.hists),
            'dumps': dc_list(parsed.dumps),
            'geoms': dc_list(parsed.geoms),
            'cracks': dc_list(parsed.cracks),
        }
        with open(args.json_out, 'w', encoding='utf-8') as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON to: {args.json_out}")


