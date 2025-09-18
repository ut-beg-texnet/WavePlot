"""Binary .MAP parser mirroring the Turbo Pascal records."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import List

from .constants import apply_var_split, func_suffix, var_name


class MapParseError(RuntimeError):
    """Raised when the .map file cannot be parsed."""


@dataclass
class SnapMapRecord:
    index: int
    offset: int
    snap_var: int
    func_id: int
    variable: str
    time_start: float
    time_end: float
    i_range: tuple[int, int]
    j_range: tuple[int, int]
    k_range: tuple[int, int]
    ax1: str
    ax2: str
    dx: float
    dy: float
    sx: int
    sy: int
    x_qty: int
    y_qty: int
    max_val: float
    min_val: float
    gnum: int
    gid: int

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start


@dataclass
class HistMapRecord:
    index: int
    offset: int
    hist_var: int
    func_id: int
    variable: str
    i_range: tuple[int, int]
    j_range: tuple[int, int]
    k_range: tuple[int, int]
    ax1: str
    dx: float
    sx: int
    sample_qty: int
    t_start: float
    t_end: float
    dt: float
    vmax: float
    gnum: int
    gid: int
    xp: int
    yp: int
    zp: int


@dataclass
class DumpMapRecord:
    index: int
    offset: int
    time: float
    i_range: tuple[int, int]
    j_range: tuple[int, int]
    k_range: tuple[int, int]
    ax1: str
    ax2: str
    dx: float
    dy: float
    sx: int
    sy: int
    i_qty: int
    j_qty: int
    v_qty: int
    gnum: int
    gid: int
    dz: float
    sz: int
    k_qty: int
    dim3: bool
    max_val: float
    x_qty: int
    y_qty: int


@dataclass
class GeomMapRecord:
    index: int
    offset: int
    gnum: int
    gid: int
    prop_total: int
    mat_total: int
    source_total: int
    stope_total: int
    time: float
    ntot: int
    i_range: tuple[int, int]
    j_range: tuple[int, int]
    k_range: tuple[int, int]
    dx: float
    dy: float
    dz: float
    grlen: int
    model3d: bool
    cog_i: int
    cog_j: int
    cog_k: int


@dataclass
class CrackMapRecord:
    index: int
    offset: int
    time: float
    ic_range: tuple[int, int]
    gnum: int
    gid: int
    st_qty: int
    t_qty: int
    v_qty: int
    cr_len: int


@dataclass
class WaveMap:
    map_path: str
    snapshots: List[SnapMapRecord] = field(default_factory=list)
    histories: List[HistMapRecord] = field(default_factory=list)
    dumps: List[DumpMapRecord] = field(default_factory=list)
    geometries: List[GeomMapRecord] = field(default_factory=list)
    crack_data: List[CrackMapRecord] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"snapshots={len(self.snapshots)}, histories={len(self.histories)}, "
            f"dumps={len(self.dumps)}, geometries={len(self.geometries)}, "
            f"cracks={len(self.crack_data)}"
        )


_FLOAT = struct.Struct('<f')
_INT = struct.Struct('<i')


def _read_float(data: memoryview, offset: int) -> tuple[float, int]:
    return _FLOAT.unpack_from(data, offset)[0], offset + 4


def _read_int(data: memoryview, offset: int) -> tuple[int, int]:
    return _INT.unpack_from(data, offset)[0], offset + 4


def _read_char(data: memoryview, offset: int) -> tuple[str, int]:
    raw = data[offset:offset + 1].tobytes()
    return raw.decode('ascii', errors='ignore') or ' ', offset + 1


def _variable_label(var_id: int, func_id: int) -> str:
    base = var_name(var_id)
    suffix = func_suffix(func_id)
    return base + (f'-{suffix}' if suffix else '') if base else f'var{var_id}'


def _read_snap_record(raw: bytes, index: int, snap_offset: int) -> tuple[SnapMapRecord, int]:
    if len(raw) != 78:
        raise MapParseError(f"Snapshot record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    snap_var_raw, pos = _read_int(data, pos)
    t1, pos = _read_float(data, pos)
    time, pos = _read_float(data, pos)
    i1, pos = _read_int(data, pos)
    i2, pos = _read_int(data, pos)
    j1, pos = _read_int(data, pos)
    j2, pos = _read_int(data, pos)
    k1, pos = _read_int(data, pos)
    k2, pos = _read_int(data, pos)
    ax1, pos = _read_char(data, pos)
    ax2, pos = _read_char(data, pos)
    dx, pos = _read_float(data, pos)
    dy, pos = _read_float(data, pos)
    sx, pos = _read_int(data, pos)
    sy, pos = _read_int(data, pos)
    x_qty, pos = _read_int(data, pos)
    y_qty, pos = _read_int(data, pos)
    max_val, pos = _read_float(data, pos)
    min_val, pos = _read_float(data, pos)
    gnum, pos = _read_int(data, pos)
    gid, pos = _read_int(data, pos)
    func_id, snap_var = apply_var_split(snap_var_raw)
    record = SnapMapRecord(
        index=index,
        offset=snap_offset,
        snap_var=snap_var,
        func_id=func_id,
        variable=_variable_label(snap_var, func_id),
        time_start=float(t1),
        time_end=float(time),
        i_range=(i1, i2),
        j_range=(j1, j2),
        k_range=(k1, k2),
        ax1=ax1,
        ax2=ax2,
        dx=float(dx),
        dy=float(dy),
        sx=sx,
        sy=sy,
        x_qty=x_qty,
        y_qty=y_qty,
        max_val=float(max_val),
        min_val=float(min_val),
        gnum=gnum,
        gid=gid,
    )
    next_offset = snap_offset + x_qty * (y_qty + 2)
    return record, next_offset


def _read_hist_record(raw: bytes, index: int, hist_offset: int) -> tuple[HistMapRecord, int]:
    if len(raw) != 65:
        raise MapParseError(f"History record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    hist_var_raw, pos = _read_int(data, pos)
    t1, pos = _read_float(data, pos)
    t2, pos = _read_float(data, pos)
    i1, pos = _read_int(data, pos)
    i2, pos = _read_int(data, pos)
    j1, pos = _read_int(data, pos)
    j2, pos = _read_int(data, pos)
    k1, pos = _read_int(data, pos)
    k2, pos = _read_int(data, pos)
    ax1, pos = _read_char(data, pos)
    dx, pos = _read_float(data, pos)
    sx, pos = _read_int(data, pos)
    x_qty, pos = _read_int(data, pos)
    max_val, pos = _read_float(data, pos)
    min_val, pos = _read_float(data, pos)
    gnum, pos = _read_int(data, pos)
    gid, pos = _read_int(data, pos)
    func_id, hist_var = apply_var_split(hist_var_raw)
    t_qty = x_qty
    xp, yp, zp = i1, j1, k1
    if ax1.lower() == 'i':
        t_start = (i1 - 1) * dx
        t_end = (i2 - 1) * dx
    elif ax1.lower() == 'j':
        t_start = (j1 - 1) * dx
        t_end = (j2 - 1) * dx
    elif ax1.lower() == 'k':
        t_start = (k1 - 1) * dx
        t_end = (k2 - 1) * dx
    else:
        t_start = t1
        t_end = t2
    if t_qty > 1:
        dt = (t_end - t_start) / (t_qty - 1) if (t_end - t_start) != 0 else dx
    else:
        dt = 0.0
    vmax = float(max(abs(max_val), abs(min_val)))
    next_offset = hist_offset + x_qty * 3
    record = HistMapRecord(
        index=index,
        offset=hist_offset,
        hist_var=hist_var,
        func_id=func_id,
        variable=_variable_label(hist_var, func_id),
        i_range=(i1, i2),
        j_range=(j1, j2),
        k_range=(k1, k2),
        ax1=ax1,
        dx=float(dx),
        sx=sx,
        sample_qty=x_qty,
        t_start=float(t_start),
        t_end=float(t_end),
        dt=float(dt),
        vmax=vmax,
        gnum=gnum,
        gid=gid,
        xp=xp,
        yp=yp,
        zp=zp,
    )
    return record, next_offset


def _read_dump_record(raw: bytes, index: int, dump_offset: int) -> tuple[DumpMapRecord, int]:
    if len(raw) != 78:
        raise MapParseError(f"Dump record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    time, pos = _read_float(data, pos)
    i1, pos = _read_int(data, pos)
    i2, pos = _read_int(data, pos)
    j1, pos = _read_int(data, pos)
    j2, pos = _read_int(data, pos)
    k1, pos = _read_int(data, pos)
    k2, pos = _read_int(data, pos)
    ax1, pos = _read_char(data, pos)
    ax2, pos = _read_char(data, pos)
    dx, pos = _read_float(data, pos)
    dy, pos = _read_float(data, pos)
    sx, pos = _read_int(data, pos)
    sy, pos = _read_int(data, pos)
    iqty, pos = _read_int(data, pos)
    jqty, pos = _read_int(data, pos)
    vqty, pos = _read_int(data, pos)
    gnum, pos = _read_int(data, pos)
    gid, pos = _read_int(data, pos)
    dz, pos = _read_float(data, pos)
    sz, pos = _read_int(data, pos)
    kqty, pos = _read_int(data, pos)
    dim3 = bool(vqty == 9)
    next_offset = dump_offset + 1 + iqty * jqty * max(kqty, 1) * vqty + 1
    record = DumpMapRecord(
        index=index,
        offset=dump_offset,
        time=float(time),
        i_range=(i1, i2),
        j_range=(j1, j2),
        k_range=(k1, k2),
        ax1=ax1,
        ax2=ax2,
        dx=float(dx),
        dy=float(dy),
        sx=sx,
        sy=sy,
        i_qty=iqty,
        j_qty=jqty,
        v_qty=vqty,
        gnum=gnum,
        gid=gid,
        dz=float(dz),
        sz=sz,
        k_qty=kqty,
        dim3=dim3,
        max_val=0.0,
        x_qty=iqty,
        y_qty=jqty,
    )
    return record, next_offset


def _read_geom_record(raw: bytes, index: int, geom_offset: int) -> tuple[GeomMapRecord, int]:
    if len(raw) != 76:
        raise MapParseError(f"Geometry record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    gnum, pos = _read_int(data, pos)
    gid, pos = _read_int(data, pos)
    prop_total_t, pos = _read_int(data, pos)
    mat_total_t, pos = _read_int(data, pos)
    source_total_t, pos = _read_int(data, pos)
    stope_total_t, pos = _read_int(data, pos)
    time, pos = _read_float(data, pos)
    ntot, pos = _read_int(data, pos)
    i1, pos = _read_int(data, pos)
    i2, pos = _read_int(data, pos)
    j1, pos = _read_int(data, pos)
    j2, pos = _read_int(data, pos)
    k1, pos = _read_int(data, pos)
    k2, pos = _read_int(data, pos)
    dx, pos = _read_float(data, pos)
    dy, pos = _read_float(data, pos)
    dz, pos = _read_float(data, pos)
    grlen, pos = _read_int(data, pos)
    # Remaining bytes represent assorted 16-bit fields and positions; treat as zero for now.
    if grlen == 0:
        grlen = (prop_total_t + mat_total_t + source_total_t + stope_total_t) * 128
    record = GeomMapRecord(
        index=index,
        offset=geom_offset,
        gnum=gnum,
        gid=gid,
        prop_total=prop_total_t,
        mat_total=mat_total_t,
        source_total=source_total_t,
        stope_total=stope_total_t,
        time=float(time),
        ntot=ntot,
        i_range=(i1, i2),
        j_range=(j1, j2),
        k_range=(k1, k2),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        grlen=grlen,
        model3d=(k2 != k1),
        cog_i=0,
        cog_j=0,
        cog_k=0,
    )
    next_offset = geom_offset + grlen
    return record, next_offset


def _read_crack_record(raw: bytes, index: int, crack_offset: int) -> tuple[CrackMapRecord, int]:
    if len(raw) != 36:
        raise MapParseError(f"Crack record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    time, pos = _read_float(data, pos)
    ic1, pos = _read_int(data, pos)
    ic2, pos = _read_int(data, pos)
    gnum, pos = _read_int(data, pos)
    gid, pos = _read_int(data, pos)
    st_qty, pos = _read_int(data, pos)
    t_qty, pos = _read_int(data, pos)
    v_qty, pos = _read_int(data, pos)
    cr_len, pos = _read_int(data, pos)
    record = CrackMapRecord(
        index=index,
        offset=crack_offset,
        time=float(time),
        ic_range=(ic1, ic2),
        gnum=gnum,
        gid=gid,
        st_qty=st_qty,
        t_qty=t_qty,
        v_qty=v_qty,
        cr_len=cr_len,
    )
    next_offset = crack_offset + cr_len
    return record, next_offset


def parse_map(map_path: str) -> WaveMap:
    """Parse the given .map file into record metadata."""
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)
    wave_map = WaveMap(map_path=os.path.abspath(map_path))
    snap_offset = dump_offset = hist_offset = geom_offset = crack_offset = 0
    with open(map_path, 'rb') as f:
        idx_snap = idx_dump = idx_hist = idx_geom = idx_crack = 0
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            # get the 'pl_type' from the header (4 bytes)
            # pl_type will be used to determine the type of record to read
            try:
                _, pl_type = struct.unpack('<ii', header)
            except struct.error as exc:
                raise MapParseError(f"Corrupt header near byte {f.tell()}") from exc
            # if pl_type is 0, we are reading a snapshot record
            if pl_type == 0:
                raw = f.read(78)
                record, snap_offset = _read_snap_record(raw, idx_snap + 1, snap_offset)
                wave_map.snapshots.append(record)
                idx_snap += 1
                f.seek(42, os.SEEK_CUR)
            # if pl_type is 1, we are reading a history record
            elif pl_type == 1:
                raw = f.read(65)
                record, hist_offset = _read_hist_record(raw, idx_hist + 1, hist_offset)
                wave_map.histories.append(record)
                idx_hist += 1
                f.seek(55, os.SEEK_CUR)
            # if pl_type is 2, we are reading a dump record
            elif pl_type == 2:
                raw = f.read(78)
                record, dump_offset = _read_dump_record(raw, idx_dump + 1, dump_offset)
                wave_map.dumps.append(record)
                idx_dump += 1
                f.seek(42, os.SEEK_CUR)
            # if pl_type is 3, we are reading a geometry record
            elif pl_type == 3:
                raw = f.read(76)
                record, geom_offset = _read_geom_record(raw, idx_geom + 1, geom_offset)
                wave_map.geometries.append(record)
                idx_geom += 1
                f.seek(44, os.SEEK_CUR)
            # if pl_type is 4, we are reading a crack record
            elif pl_type == 4:
                raw = f.read(36)
                record, crack_offset = _read_crack_record(raw, idx_crack + 1, crack_offset)
                wave_map.crack_data.append(record)
                idx_crack += 1
                f.seek(84, os.SEEK_CUR)
            # if pl_type is not 0-4, throw an error
            else:
                raise MapParseError(f"Unknown record type {pl_type} at byte {f.tell()-8}")
    return wave_map
