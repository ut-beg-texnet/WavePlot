"""
.MAP reader utilities for WavePlot

Notes
- The original Turbo Pascal code (DATA_RSC.PAS) stores map entries using a
  simple header followed by a fixed-size body per entry. Each entry begins
  with two 4‑byte little-endian integers:
    1) a dummy integer (ignored)
    2) the entry type (PlType):
       0 = snapshot, 1 = history, 2 = dump, 3 = geometry, 4 = crack data
  After the header, there is a body (120 bytes).

- DATA_RSC.PAS selectively reads 65/76/78 bytes (then seeks further) from the
  body depending on PlType. For robustness we read the full 120‑byte body and
  decode only the leading fields that are consistent across versions.

- The extracted "dt" is computed from the first History record as
  dt ≈ (t2 - t1) / (Tqty - 1), where Tqty is the X‑quantity in that record.
  Grid extents and spacings are taken from the first Geometry record as
  I = i2-i1+1, J = j2-j1+1, K = k2-k1+1 with spacings (dx,dy,dz).

This module avoids fully defining all binary layouts on purpose. When we
need more detailed metadata (axes semantics, variable IDs, offsets, etc.) we
can extend this reader while referencing DATA_RSC.PAS and the reader in
"Turbo pascal version/read_map.py".
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Dict


# Fixed size of each record payload after the 8‑byte header (see DATA_RSC.PAS
# where each branch reads a subset then seeks to align to the next record)
MAP_RECORD_BODY_BYTES = 120


@dataclass
class MapSummary:
    snapshots: int
    histories: int
    dumps: int
    geometries: int
    crack_data: int

    @property
    def total_records(self) -> int:
        return self.snapshots + self.histories + self.dumps + self.geometries + self.crack_data


@dataclass
class MapQuickMeta:
    """Optional quick metadata extracted from the first History/Geometry entries.

    All fields are optional and may be None if not derivable from the file.
    """
    # History-derived timing
    t_start: Optional[float]
    t_end: Optional[float]
    t_qty: Optional[int]
    dt: Optional[float]

    # Geometry-derived grid extents and spacing
    i_dim: Optional[int]
    j_dim: Optional[int]
    k_dim: Optional[int]
    dx: Optional[float]
    dy: Optional[float]
    dz: Optional[float]


# returns the absolute path without extension for a given .map file path
def map_to_base_filename(map_path: str) -> str:
    return os.path.splitext(os.path.abspath(map_path))[0]


def base_to_related_paths(base: str) -> dict:
    """
    Return likely related file paths for a given base (no extension).
    We assume that the related files have the same base name (only extension is different).
    """
    return {
        "map": base + ".map",
        "hst": base + ".hst",
        "m": base + ".m",
        "snp": base + ".snp",
        "dmp": base + ".dmp",
        "geo": base + ".geo",
    }


# ---------------- Variable name mapping (from DATA_RSC.PAS) -----------------

# Section counts
_ngrvar = 10; _ngcvar = 21; _nstvar = 21; _nscvar = 11
_ngmxvar = 1; _nsmxvar = 8; _ngacvar = 3; _nsacvar = 1; _nmixvar = 4

# Section offsets
_grnam0 = 0; _gcnam0 = 20; _stnam0 = 50; _scnam0 = 100
_gmxnam0 = 150; _smxnam0 = 175; _gacnam0 = 200; _sacnam0 = 225; _mixnam0 = 235

# Function names for HFunc (index 0..5 in Pascal; 0 is empty)
_VFUNC = ['', 'MX', 'mn', 'Avg', 'Abs', 'AMX']

# Variable name tables (taken from DATA_RSC.PAS)
_GRVAR = ['XVEL','YVEL','S11','S22','S12','ZVEL','S33','S23','S31','S31']
_GCVAR = ['DIL','V-abs','TauMx','Sig-3','Sig-1','Sig-2','M-Ang1','ESS','DEV',
          'aXVEL','aYVEL','aS12','aZVEL','aS23','aS31','aS31','S.E.','K.E.',
          'T.E.','Fail','e-plas']
_STVAR = ['t11_s1','t11_s0','t1v_s1','t1v_s0','t1sh','t1d-r',
          'nvel-r','ndis-r','Sh-bnd','T1slip','TaSlip','N-bnd',
          't22_s1','t22_s0','t2v_s1','t2v_s0','t1t2s1','t1t2s0','t2sh','t2d-r','T2slip']
_SCVAR = ['nd-s1','nd-s0','Tad-r','t1d_s1','t1d_s0','t2d_s1','t2d_s0',
          'stv_s1','stv_s0','sts_s1','sts_s0']
_GMXVAR = ['MaxVel']
_SMXVAR = ['nv-mx','nd-mx','t1v-mx','t1d-mx','t2v-mx','t2d-mx','t11-mx','t22-mx']
_GACVAR = ['Xdisp','Ydisp','Zdisp']
_SACVAR = ['']
_MIXVAR = ['skn','sks','sNs','sSs']


def _var_name(vnum: int) -> str:
    """Python port of VarName from DATA_RSC.PAS."""
    if vnum <= 0:
        return ''
    if (_grnam0+1) <= vnum <= _gcnam0:
        return _GRVAR[vnum-_grnam0-1]
    if (_gcnam0+1) <= vnum <= _stnam0:
        return _GCVAR[vnum-_gcnam0-1]
    if (_stnam0+1) <= vnum <= _scnam0:
        return _STVAR[vnum-_stnam0-1]
    if (_scnam0+1) <= vnum <= _gmxnam0:
        return _SCVAR[vnum-_scnam0-1]
    if (_gmxnam0+1) <= vnum <= _smxnam0:
        return _GMXVAR[vnum-_gmxnam0-1]
    if (_smxnam0+1) <= vnum <= _gacnam0:
        return _SMXVAR[vnum-_smxnam0-1]
    if (_gacnam0+1) <= vnum <= _sacnam0:
        return _GACVAR[vnum-_gacnam0-1]
    if (_sacnam0+1) <= vnum <= _mixnam0:
        return _SACVAR[vnum-_sacnam0-1]
    if (_mixnam0+1) <= vnum <= 254:
        return _MIXVAR[vnum-_mixnam0-1]
    if vnum == 255:
        return 'Form'
    return ''


def history_labels_from_map(map_path: str) -> List[str]:
    """Return human-readable labels for each history record (pl_type==1).

    The label is built from VarName(Vnum) plus the function suffix derived from
    HFunc (embedded in HistVar as in DATA_RSC.PAS). If multiple histories share
    the same base name, a running index is appended to make labels unique.
    """
    labels: List[str] = []
    if not os.path.exists(map_path):
        return labels
    counts: Dict[str, int] = {}
    with open(map_path, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            try:
                _, pl_type = struct.unpack('<II', header)
            except struct.error:
                break

            body = f.read(MAP_RECORD_BODY_BYTES)
            if len(body) < MAP_RECORD_BODY_BYTES:
                break

            if pl_type == 1:
                try:
                    histvar_raw = struct.unpack('<l', body[0:4])[0]
                    # Unpack combined HistVar (low 16 bits) and HFunc (div by 0xFFFF) like Pascal
                    hfunc = histvar_raw // 0xFFFF
                    vnum = histvar_raw % 0xFFFF
                    base = _var_name(vnum) or f'var{vnum}'
                    suffix = _VFUNC[hfunc] if 0 <= hfunc < len(_VFUNC) else ''
                    name = base + (f'-{suffix}' if suffix else '')
                except Exception:
                    name = 'hist'

                # Make unique
                counts[name] = counts.get(name, 0) + 1
                idx = counts[name]
                label = f"{name}_{idx}" if idx > 1 else name
                labels.append(label)

    return labels


def read_map_summary(map_path: str) -> MapSummary:
    """
    Read the .map file and return a summary of contained record types.

    This only parses the 8‑byte header for each entry and skips over the fixed
    120‑byte body. It is resilient to extra bytes at the end of the file and
    partial trailing entries.
    """
    if not os.path.exists(map_path):
        raise FileNotFoundError(f".map file not found: {map_path}")

    snapshots = 0
    histories = 0
    dumps = 0
    geometries = 0
    crack_data = 0

    with open(map_path, "rb") as f:
        while True:
            header = f.read(8) # read the 8-byte header
            if len(header) < 8:
                break  # EOF reached
            try:
                # Little‑endian unsigned ints: (ignored_dword, pl_type)
                _, pl_type = struct.unpack("<II", header) # unpacks the 8-byte header into two unsigned integers (4 bytes each)
            except struct.error:
                break  # Trailing bytes; stop safely

            # Count entry types by pl_type
            if pl_type == 0:
                snapshots += 1
            elif pl_type == 1:
                histories += 1
            elif pl_type == 2:
                dumps += 1
            elif pl_type == 3:
                geometries += 1
            elif pl_type == 4:
                crack_data += 1

            # Always read/skip the fixed body so the stream aligns correctly
            body = f.read(MAP_RECORD_BODY_BYTES)
            if len(body) < MAP_RECORD_BODY_BYTES:
                break

    return MapSummary(
        snapshots=snapshots,
        histories=histories,
        dumps=dumps,
        geometries=geometries,
        crack_data=crack_data,
    )


def read_quick_meta(map_path: str) -> MapQuickMeta:
    """Parse the first History and Geometry entries to estimate dt and grid dims.

    This uses conservative slicing of the 120‑byte bodies based on DATA_RSC.PAS
    and our experimental reader. If parsing fails, fields remain None.
    """
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    t_qty: Optional[int] = None
    dt: Optional[float] = None

    i_dim: Optional[int] = None
    j_dim: Optional[int] = None
    k_dim: Optional[int] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    dz: Optional[float] = None

    try:
        with open(map_path, "rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                try:
                    _, pl_type = struct.unpack("<II", header)
                except struct.error:
                    break

                body = f.read(MAP_RECORD_BODY_BYTES)
                if len(body) < MAP_RECORD_BODY_BYTES:
                    break

                # First History: derive dt from t1,t2 and Xqty
                if pl_type == 1 and dt is None:
                    try:
                        # Layout based on Turbo Pascal DI_ReadMap (65 bytes read there)
                        # Here we read from the start of the 120‑byte body.
                        # hist_var (0:4), t1 (4:8), t2 (8:12)
                        t1_val, t2_val = struct.unpack("<ff", body[4:12])
                        x_qty_val = struct.unpack("<l", body[45:49])[0]
                        if x_qty_val and x_qty_val > 1:
                            t_start = float(t1_val)
                            t_end = float(t2_val)
                            t_qty = int(x_qty_val)
                            dt = (t_end - t_start) / (t_qty - 1)
                    except Exception:
                        pass

                # First Geometry: get i1..k2 and dx,dy,dz
                if pl_type == 3 and (i_dim is None or dx is None):
                    try:
                        # Parse 72 bytes worth of leading fields (see DATA_RSC.PAS GeomMapRec)
                        # gnum,g_id,proptotT,mattotT,sourcetotT,stopetotT,time,ntot,
                        # i1,i2,j1,j2,k1,k2,dx,dy,dz,grlen
                        fmt = "<llllllfllllllllfff l"
                        # Python's struct does not allow spaces; construct stepwise
                        fmt = "<" + "l"*6 + "f" + "l" + "l"*6 + "f"*3 + "l"
                        (gnum, g_id, proptot_t, mattot_t, sourcetot_t, stopetot_t,
                         time_val, ntot,
                         i1, i2, j1, j2, k1, k2,
                         dx_val, dy_val, dz_val, grlen) = struct.unpack(fmt, body[:72])
                        # Compute dims if sensible
                        def dim(a: int, b: int) -> Optional[int]:
                            try:
                                val = int(b) - int(a) + 1
                                return val if val > 0 else None
                            except Exception:
                                return None
                        i_dim = i_dim or dim(i1, i2)
                        j_dim = j_dim or dim(j1, j2)
                        k_dim = k_dim or dim(k1, k2)
                        dx = dx if dx is not None else float(dx_val)
                        dy = dy if dy is not None else float(dy_val)
                        dz = dz if dz is not None else float(dz_val)
                    except Exception:
                        pass

                # Stop early if we've collected both
                if (dt is not None) and (dx is not None) and (i_dim is not None):
                    break
    except FileNotFoundError:
        pass

    return MapQuickMeta(
        t_start=t_start, t_end=t_end, t_qty=t_qty, dt=dt,
        i_dim=i_dim, j_dim=j_dim, k_dim=k_dim, dx=dx, dy=dy, dz=dz,
    )


def try_resolve_hst_from_map(map_path: str) -> Optional[str]:
    """
    Given a .map file, return the path to the corresponding .hst file if it
    exists (same base name, different extension). Returns None if not found.
    """
    base = map_to_base_filename(map_path)
    hst_path = base + ".hst"
    if os.path.exists(hst_path):
        return hst_path
    return None


def try_resolve_m_from_map(map_path: str) -> Optional[str]:
    """Return matching .m path for a map if it exists, else None."""
    base = map_to_base_filename(map_path)
    m_path = base + ".m"
    if os.path.exists(m_path):
        return m_path
    return None


if __name__ == "__main__":
    # Simple CLI test when running this file directly
    import argparse

    parser = argparse.ArgumentParser(description="Read a .map file and print a summary")
    parser.add_argument("--map-file", type=str, help="Path to the .map file")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file to write the summary to. If not provided, the summary will be printed to the console.")
    args = parser.parse_args()

    summary = read_map_summary(args.map_file)
    meta = read_quick_meta(args.map_file)
    if args.output:
        with open(args.output, "w") as f:
            f.write(".MAP summary:\n")
            f.write(f"  Snapshots : {summary.snapshots}\n")
            f.write(f"  Histories : {summary.histories}\n")
            f.write(f"  Dumps     : {summary.dumps}\n")
            f.write(f"  Geometries: {summary.geometries}\n")
            f.write(f"  CrackData : {summary.crack_data}\n")

            if any(v is not None for v in [meta.dt, meta.t_qty, meta.t_start, meta.t_end]):
                f.write("\nHistory timing (from first history record):\n")
                if meta.t_qty is not None:
                    f.write(f"  Samples   : {meta.t_qty}\n")
                if meta.t_start is not None and meta.t_end is not None:
                    f.write(f"  Range     : {meta.t_start} .. {meta.t_end}\n")
                if meta.dt is not None:
                    f.write(f"  Time step : {meta.dt}\n")

            if any(v is not None for v in [meta.i_dim, meta.j_dim, meta.k_dim, meta.dx, meta.dy, meta.dz]):
                f.write("\nGrid (from first geometry record):\n")
                dims = [meta.i_dim, meta.j_dim, meta.k_dim]
                if any(v is not None for v in dims):
                    f.write(f"  Dimensions: I={meta.i_dim}  J={meta.j_dim}  K={meta.k_dim}\n")
                if any(v is not None for v in [meta.dx, meta.dy, meta.dz]):
                    f.write(f"  Spacing   : dx={meta.dx}  dy={meta.dy}  dz={meta.dz}\n")
    else:
        print(".MAP summary:")
        print(f"  Snapshots : {summary.snapshots}")
        print(f"  Histories : {summary.histories}")
        print(f"  Dumps     : {summary.dumps}")
        print(f"  Geometries: {summary.geometries}")
        print(f"  CrackData : {summary.crack_data}")

        # Optional quick metadata (History timing and Geometry grid)
        if any(v is not None for v in [meta.dt, meta.t_qty, meta.t_start, meta.t_end]):
            print("\nHistory timing (from first history record):")
            if meta.t_qty is not None:
                print(f"  Samples   : {meta.t_qty}")
            if meta.t_start is not None and meta.t_end is not None:
                print(f"  Range     : {meta.t_start} .. {meta.t_end}")
            if meta.dt is not None:
                print(f"  Time step : {meta.dt}")

        if any(v is not None for v in [meta.i_dim, meta.j_dim, meta.k_dim, meta.dx, meta.dy, meta.dz]):
            print("\nGrid (from first geometry record):")
            dims = [meta.i_dim, meta.j_dim, meta.k_dim]
            if any(v is not None for v in dims):
                print(f"  Dimensions: I={meta.i_dim}  J={meta.j_dim}  K={meta.k_dim}")
            if any(v is not None for v in [meta.dx, meta.dy, meta.dz]):
                print(f"  Spacing   : dx={meta.dx}  dy={meta.dy}  dz={meta.dz}")

    base = map_to_base_filename(args.map_file)
    paths = base_to_related_paths(base)
    '''
    print("\nRelated paths (if present):")
    for k, v in paths.items():
        print(f"  {k:>3}: {v}  {'(found)' if os.path.exists(v) else '(missing)'}")
    '''

