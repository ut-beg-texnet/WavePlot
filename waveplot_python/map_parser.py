"""Binary .MAP parser mirroring the Turbo Pascal records.

This module parses the binary .map file format used by the Turbo Pascal WAVE program.
The .map file contains metadata records that describe the structure and location
of data in companion binary files (.snp, .hst, .dmp, .geo, .crk).

File Format:
- Each record consists of an 8-byte header followed by a 120-byte body
- Header: (dummy_int, pl_type) where pl_type indicates record type:
  * 0 = Snapshot
  * 1 = History
  * 2 = Dump
  * 3 = Geometry
  * 4 = Crack data
- Body: Variable-length field section + padding (only leading bytes used per type)
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import List

from .constants import apply_var_split, func_suffix, var_name


class MapParseError(RuntimeError):
    """Exception raised when the .map file cannot be parsed due to corruption or invalid format."""


@dataclass
class SnapMapRecord:
    """Snapshot record metadata parsed from .map file.
    
    Snapshots are 2D spatial data arrays representing a field variable at a specific
    time or time range. The data is stored in the .snp file starting at the given offset.
    
    Attributes:
        index: 1-based record index
        offset: File offset in float32 units for reading from .snp file
        snap_var: Variable ID (low 16 bits of combined variable/function integer)
        func_id: Function ID (high 16 bits, e.g., 1=MX, 3=Avg)
        variable: Human-readable variable name with function suffix
        time_start: Start time of snapshot window (seconds)
        time_end: End time of snapshot window (seconds)
        i_range: (i1, i2) grid range in i-direction
        j_range: (j1, j2) grid range in j-direction
        k_range: (k1, k2) grid range in k-direction
        ax1: First axis character ('i', 'j', or 'k')
        ax2: Second axis character ('i', 'j', or 'k')
        dx: Grid spacing in ax1 direction
        dy: Grid spacing in ax2 direction
        sx: Starting index offset in ax1 direction
        sy: Starting index offset in ax2 direction
        x_qty: Number of points in x/ax1 direction
        y_qty: Number of points in y/ax2 direction
        max_val: Maximum value in the snapshot
        min_val: Minimum value in the snapshot
        gnum: Geometry number identifier
        gid: Geometry ID identifier
    """
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
        """Calculate the time duration covered by this snapshot."""
        return self.time_end - self.time_start


@dataclass
class HistMapRecord:
    """History record metadata parsed from .map file.
    
    Histories are 1D time series data representing a field variable at a specific
    spatial location over time. The data is stored in the .hst file starting at offset.
    
    Attributes:
        index: 1-based record index
        offset: File offset in float32 units for reading from .hst file
        hist_var: Variable ID (low 16 bits of combined variable/function integer)
        func_id: Function ID (high 16 bits)
        variable: Human-readable variable name with function suffix
        i_range: (i1, i2) grid range in i-direction
        j_range: (j1, j2) grid range in j-direction
        k_range: (k1, k2) grid range in k-direction
        ax1: Axis character ('i', 'j', or 'k') along which time varies
        dx: Grid spacing (or time step if ax1 represents time)
        sx: Starting index offset
        sample_qty: Number of time samples
        t_start: Start time (seconds)
        t_end: End time (seconds)
        dt: Time step between samples (seconds)
        vmax: Maximum absolute value in the history
        gnum: Geometry number identifier
        gid: Geometry ID identifier
        xp: X position index (i1)
        yp: Y position index (j1)
        zp: Z position index (k1)
    """
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
    """Dump record metadata parsed from .map file.
    
    Dumps are 3D or 2D volume data arrays containing multiple variables at a
    specific time. The data is stored in the .dmp file starting at offset.
    
    Attributes:
        index: 1-based record index
        offset: File offset in float32 units for reading from .dmp file
        time: Time of the dump (seconds)
        i_range: (i1, i2) grid range in i-direction
        j_range: (j1, j2) grid range in j-direction
        k_range: (k1, k2) grid range in k-direction
        ax1: First axis character ('i', 'j', or 'k')
        ax2: Second axis character ('i', 'j', or 'k')
        dx: Grid spacing in ax1 direction
        dy: Grid spacing in ax2 direction
        sx: Starting index offset in ax1 direction
        sy: Starting index offset in ax2 direction
        i_qty: Number of grid points in i-direction
        j_qty: Number of grid points in j-direction
        v_qty: Number of variables per grid point (9 for 3D, other for 2D)
        gnum: Geometry number identifier
        gid: Geometry ID identifier
        dz: Grid spacing in k-direction
        sz: Starting index offset in k-direction
        k_qty: Number of grid points in k-direction
        dim3: True if 3D (v_qty == 9), False if 2D
        max_val: Maximum value (computed, may be 0.0)
        x_qty: Alias for i_qty
        y_qty: Alias for j_qty
    """
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
    """Geometry record metadata parsed from .map file.
    
    Geometry records describe the structure and location of geometry data in the .geo file.
    Geometry includes material properties, material regions, sources, and stopes.
    
    Attributes:
        index: 1-based record index
        offset: File offset in bytes for reading from .geo file
        gnum: Geometry number identifier
        gid: Geometry ID identifier
        prop_total: Number of property records
        mat_total: Number of material region records
        source_total: Number of source region records
        stope_total: Number of stope records
        time: Time associated with this geometry (seconds)
        ntot: Total number of nodes/elements
        i_range: (i1, i2) grid range in i-direction
        j_range: (j1, j2) grid range in j-direction
        k_range: (k1, k2) grid range in k-direction
        dx: Grid spacing in i-direction
        dy: Grid spacing in j-direction
        dz: Grid spacing in k-direction
        grlen: Length of geometry data block in bytes
        model3d: True if 3D model (k2 != k1), False if 2D
        cog_i: Center of gravity i-coordinate (computed, may be 0)
        cog_j: Center of gravity j-coordinate (computed, may be 0)
        cog_k: Center of gravity k-coordinate (computed, may be 0)
    """
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
    """Crack data record metadata parsed from .map file.
    
    Crack records describe fracture/crack data stored in the .crk file.
    
    Attributes:
        index: 1-based record index
        offset: File offset in bytes for reading from .crk file
        time: Time of the crack data (seconds)
        ic_range: (ic1, ic2) crack index range
        gnum: Geometry number identifier
        gid: Geometry ID identifier
        st_qty: Number of stations
        t_qty: Number of time points
        v_qty: Number of variables
        cr_len: Length of crack data block in bytes
    """
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
    """Complete parsed .map file containing all record metadata.
    
    This is the main container class that holds all parsed records from a .map file.
    It provides access to snapshots, histories, dumps, geometries, and crack data.
    
    Attributes:
        map_path: Absolute path to the .map file
        snapshots: List of snapshot records
        histories: List of history records
        dumps: List of dump records
        geometries: List of geometry records
        crack_data: List of crack data records
    """
    map_path: str
    snapshots: List[SnapMapRecord] = field(default_factory=list)
    histories: List[HistMapRecord] = field(default_factory=list)
    dumps: List[DumpMapRecord] = field(default_factory=list)
    geometries: List[GeomMapRecord] = field(default_factory=list)
    crack_data: List[CrackMapRecord] = field(default_factory=list)

    def summary(self) -> str:
        """Return a summary string of record counts.
        
        Returns:
            String describing the number of each record type
        """
        return (
            f"snapshots={len(self.snapshots)}, histories={len(self.histories)}, "
            f"dumps={len(self.dumps)}, geometries={len(self.geometries)}, "
            f"cracks={len(self.crack_data)}"
        )


# Pre-compiled struct objects for efficient binary parsing (little-endian)
_FLOAT = struct.Struct('<f')  # Little-endian float32
_INT = struct.Struct('<i')    # Little-endian int32


def _read_float(data: memoryview, offset: int) -> tuple[float, int]:
    """Read a float32 value from a memoryview and advance offset.
    
    Args:
        data: Memoryview of binary data
        offset: Starting byte offset
        
    Returns:
        Tuple of (float_value, new_offset)
    """
    return _FLOAT.unpack_from(data, offset)[0], offset + 4


def _read_int(data: memoryview, offset: int) -> tuple[int, int]:
    """Read an int32 value from a memoryview and advance offset.
    
    Args:
        data: Memoryview of binary data
        offset: Starting byte offset
        
    Returns:
        Tuple of (int_value, new_offset)
    """
    return _INT.unpack_from(data, offset)[0], offset + 4


def _read_char(data: memoryview, offset: int) -> tuple[str, int]:
    """Read a single ASCII character from a memoryview and advance offset.
    
    Args:
        data: Memoryview of binary data
        offset: Starting byte offset
        
    Returns:
        Tuple of (char_string, new_offset). Returns ' ' if decode fails.
    """
    raw = data[offset:offset + 1].tobytes()
    return raw.decode('ascii', errors='ignore') or ' ', offset + 1


def _variable_label(var_id: int, func_id: int) -> str:
    """Construct a human-readable variable label with optional function suffix.
    
    Args:
        var_id: Variable ID number
        func_id: Function ID number (0 = no function)
        
    Returns:
        Variable name string, e.g., "XVEL-MX" or "S11"
    """
    base = var_name(var_id)
    suffix = func_suffix(func_id)
    return base + (f'-{suffix}' if suffix else '') if base else f'var{var_id}'


def _read_snap_record(raw: bytes, index: int, snap_offset: int) -> tuple[SnapMapRecord, int]:
    """Parse a snapshot record from binary data.
    
    Snapshot records are 78 bytes. This function parses all fields and computes
    the next file offset for subsequent records.
    
    Args:
        raw: 78-byte binary record data (excluding 8-byte header)
        index: 1-based record index
        snap_offset: Current file offset in float32 units
        
    Returns:
        Tuple of (SnapMapRecord, next_offset) where next_offset is the file offset
        for the next snapshot record
        
    Raises:
        MapParseError: If record is truncated or invalid
    """
    if len(raw) != 78:
        raise MapParseError(f"Snapshot record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    # Parse fields in order (matches Turbo Pascal record layout)
    snap_var_raw, pos = _read_int(data, pos)  # Combined var/func ID
    t1, pos = _read_float(data, pos)          # Start time
    time, pos = _read_float(data, pos)         # End time
    i1, pos = _read_int(data, pos)            # i-range start
    i2, pos = _read_int(data, pos)            # i-range end
    j1, pos = _read_int(data, pos)            # j-range start
    j2, pos = _read_int(data, pos)            # j-range end
    k1, pos = _read_int(data, pos)            # k-range start
    k2, pos = _read_int(data, pos)            # k-range end
    ax1, pos = _read_char(data, pos)          # First axis
    ax2, pos = _read_char(data, pos)          # Second axis
    dx, pos = _read_float(data, pos)          # Grid spacing ax1
    dy, pos = _read_float(data, pos)          # Grid spacing ax2
    sx, pos = _read_int(data, pos)            # Start offset ax1
    sy, pos = _read_int(data, pos)            # Start offset ax2
    x_qty, pos = _read_int(data, pos)          # Points in x/ax1
    y_qty, pos = _read_int(data, pos)          # Points in y/ax2
    max_val, pos = _read_float(data, pos)     # Maximum value
    min_val, pos = _read_float(data, pos)     # Minimum value
    gnum, pos = _read_int(data, pos)          # Geometry number
    gid, pos = _read_int(data, pos)           # Geometry ID
    
    # Split combined variable/function ID
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
    # Compute next offset: snapshots include 2 padding columns (y_qty + 2)
    next_offset = snap_offset + x_qty * (y_qty + 2)
    return record, next_offset


def _read_hist_record(raw: bytes, index: int, hist_offset: int) -> tuple[HistMapRecord, int]:
    """Parse a history record from binary data.
    
    History records are 65 bytes. This function parses all fields, computes time
    information based on the axis, and calculates the next file offset.
    
    Args:
        raw: 65-byte binary record data (excluding 8-byte header)
        index: 1-based record index
        hist_offset: Current file offset in float32 units
        
    Returns:
        Tuple of (HistMapRecord, next_offset) where next_offset is the file offset
        for the next history record
        
    Raises:
        MapParseError: If record is truncated or invalid
    """
    if len(raw) != 65:
        raise MapParseError(f"History record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    # Parse fields in order
    hist_var_raw, pos = _read_int(data, pos)   # Combined var/func ID
    t1, pos = _read_float(data, pos)           # Time field 1
    t2, pos = _read_float(data, pos)           # Time field 2
    i1, pos = _read_int(data, pos)             # i-range start
    i2, pos = _read_int(data, pos)             # i-range end
    j1, pos = _read_int(data, pos)             # j-range start
    j2, pos = _read_int(data, pos)             # j-range end
    k1, pos = _read_int(data, pos)             # k-range start
    k2, pos = _read_int(data, pos)             # k-range end
    ax1, pos = _read_char(data, pos)           # Axis along which time varies
    dx, pos = _read_float(data, pos)           # Grid spacing or time step
    sx, pos = _read_int(data, pos)             # Start offset
    x_qty, pos = _read_int(data, pos)          # Number of samples
    max_val, pos = _read_float(data, pos)      # Maximum value
    min_val, pos = _read_float(data, pos)      # Minimum value
    gnum, pos = _read_int(data, pos)           # Geometry number
    gid, pos = _read_int(data, pos)            # Geometry ID
    
    # Split combined variable/function ID
    func_id, hist_var = apply_var_split(hist_var_raw)
    
    # Compute time range based on axis
    # If ax1 is 'i', 'j', or 'k', time is computed from grid positions
    # Otherwise, use the t1/t2 values directly
    t_qty = x_qty
    xp, yp, zp = i1, j1, k1  # Position indices
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
    
    # Compute time step
    if t_qty > 1:
        dt = (t_end - t_start) / (t_qty - 1) if (t_end - t_start) != 0 else dx
    else:
        dt = 0.0
    
    # Maximum absolute value
    vmax = float(max(abs(max_val), abs(min_val)))
    
    # Compute next offset: histories store 3 floats per sample
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
    """Parse a dump record from binary data.
    
    Dump records are 78 bytes. This function parses all fields and computes
    the next file offset. Dumps include sentinel values (1 float before and after data).
    
    Args:
        raw: 78-byte binary record data (excluding 8-byte header)
        index: 1-based record index
        dump_offset: Current file offset in float32 units
        
    Returns:
        Tuple of (DumpMapRecord, next_offset) where next_offset is the file offset
        for the next dump record
        
    Raises:
        MapParseError: If record is truncated or invalid
    """
    if len(raw) != 78:
        raise MapParseError(f"Dump record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    # Parse fields in order
    time, pos = _read_float(data, pos)         # Time of dump
    i1, pos = _read_int(data, pos)           # i-range start
    i2, pos = _read_int(data, pos)            # i-range end
    j1, pos = _read_int(data, pos)            # j-range start
    j2, pos = _read_int(data, pos)            # j-range end
    k1, pos = _read_int(data, pos)            # k-range start
    k2, pos = _read_int(data, pos)            # k-range end
    ax1, pos = _read_char(data, pos)          # First axis
    ax2, pos = _read_char(data, pos)          # Second axis
    dx, pos = _read_float(data, pos)          # Grid spacing ax1
    dy, pos = _read_float(data, pos)          # Grid spacing ax2
    sx, pos = _read_int(data, pos)            # Start offset ax1
    sy, pos = _read_int(data, pos)            # Start offset ax2
    iqty, pos = _read_int(data, pos)          # Points in i-direction
    jqty, pos = _read_int(data, pos)          # Points in j-direction
    vqty, pos = _read_int(data, pos)          # Variables per point
    gnum, pos = _read_int(data, pos)          # Geometry number
    gid, pos = _read_int(data, pos)           # Geometry ID
    dz, pos = _read_float(data, pos)          # Grid spacing k-direction
    sz, pos = _read_int(data, pos)            # Start offset k-direction
    kqty, pos = _read_int(data, pos)          # Points in k-direction
    
    # Determine if 3D (v_qty == 9 indicates 3D dump)
    dim3 = bool(vqty == 9)
    
    # Compute next offset: dumps have sentinel values (1 float before + after data)
    # Data size: iqty * jqty * kqty * vqty floats
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
        max_val=0.0,  # Computed elsewhere if needed
        x_qty=iqty,
        y_qty=jqty,
    )
    return record, next_offset


def _read_geom_record(raw: bytes, index: int, geom_offset: int) -> tuple[GeomMapRecord, int]:
    """Parse a geometry record from binary data.
    
    Geometry records are 76 bytes. This function parses all fields and computes
    the next file offset. If grlen is 0, it's computed from record counts.
    
    Args:
        raw: 76-byte binary record data (excluding 8-byte header)
        index: 1-based record index
        geom_offset: Current file offset in bytes
        
    Returns:
        Tuple of (GeomMapRecord, next_offset) where next_offset is the file offset
        for the next geometry record
        
    Raises:
        MapParseError: If record is truncated or invalid
    """
    if len(raw) != 76:
        raise MapParseError(f"Geometry record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    # Parse fields in order
    gnum, pos = _read_int(data, pos)           # Geometry number
    gid, pos = _read_int(data, pos)            # Geometry ID
    prop_total_t, pos = _read_int(data, pos)   # Property count
    mat_total_t, pos = _read_int(data, pos)    # Material count
    source_total_t, pos = _read_int(data, pos) # Source count
    stope_total_t, pos = _read_int(data, pos)  # Stope count
    time, pos = _read_float(data, pos)         # Time
    ntot, pos = _read_int(data, pos)           # Total nodes
    i1, pos = _read_int(data, pos)             # i-range start
    i2, pos = _read_int(data, pos)             # i-range end
    j1, pos = _read_int(data, pos)             # j-range start
    j2, pos = _read_int(data, pos)             # j-range end
    k1, pos = _read_int(data, pos)             # k-range start
    k2, pos = _read_int(data, pos)             # k-range end
    dx, pos = _read_float(data, pos)           # Grid spacing i
    dy, pos = _read_float(data, pos)           # Grid spacing j
    dz, pos = _read_float(data, pos)           # Grid spacing k
    grlen, pos = _read_int(data, pos)          # Geometry data length
    
    # If grlen is 0, compute it from record counts (each record is 128 bytes)
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
        model3d=(k2 != k1),  # 3D if k-range spans more than one point
        cog_i=0,  # Center of gravity computed elsewhere if needed
        cog_j=0,
        cog_k=0,
    )
    # Next offset is current offset plus geometry data length
    next_offset = geom_offset + grlen
    return record, next_offset


def _read_crack_record(raw: bytes, index: int, crack_offset: int) -> tuple[CrackMapRecord, int]:
    """Parse a crack data record from binary data.
    
    Crack records are 36 bytes. This function parses all fields and computes
    the next file offset.
    
    Args:
        raw: 36-byte binary record data (excluding 8-byte header)
        index: 1-based record index
        crack_offset: Current file offset in bytes
        
    Returns:
        Tuple of (CrackMapRecord, next_offset) where next_offset is the file offset
        for the next crack record
        
    Raises:
        MapParseError: If record is truncated or invalid
    """
    if len(raw) != 36:
        raise MapParseError(f"Crack record {index} truncated (got {len(raw)} bytes)")
    data = memoryview(raw)
    pos = 0
    # Parse fields in order
    time, pos = _read_float(data, pos)      # Time of crack data
    ic1, pos = _read_int(data, pos)        # Crack index range start
    ic2, pos = _read_int(data, pos)        # Crack index range end
    gnum, pos = _read_int(data, pos)       # Geometry number
    gid, pos = _read_int(data, pos)        # Geometry ID
    st_qty, pos = _read_int(data, pos)     # Number of stations
    t_qty, pos = _read_int(data, pos)      # Number of time points
    v_qty, pos = _read_int(data, pos)      # Number of variables
    cr_len, pos = _read_int(data, pos)     # Crack data length in bytes
    
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
    # Next offset is current offset plus crack data length
    next_offset = crack_offset + cr_len
    return record, next_offset


def parse_map(map_path: str) -> WaveMap:
    """Parse a .map file and extract all record metadata.
    
    This is the main entry point for parsing .map files. It reads the file sequentially,
    parsing each record based on its type (determined by pl_type in the header).
    File offsets are tracked separately for each record type since they reference
    different binary files (.snp, .hst, .dmp, .geo, .crk).
    
    File Structure:
    - Each record: 8-byte header + variable-length body + padding to 128 bytes total
    - Header: (dummy_int, pl_type) where pl_type indicates record type
    - Body length varies by type: snapshots=78, histories=65, dumps=78, geometries=76, cracks=36
    - Remaining bytes are padding (skipped)
    
    Args:
        map_path: Path to the .map file to parse
        
    Returns:
        WaveMap object containing all parsed records
        
    Raises:
        FileNotFoundError: If the map file doesn't exist
        MapParseError: If the file is corrupted or has invalid records
    """
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)
    
    wave_map = WaveMap(map_path=os.path.abspath(map_path))
    # Track file offsets separately for each record type (they reference different files)
    snap_offset = dump_offset = hist_offset = geom_offset = crack_offset = 0
    
    with open(map_path, 'rb') as f:
        # Track indices for each record type (1-based)
        idx_snap = idx_dump = idx_hist = idx_geom = idx_crack = 0
        
        while True:
            # Read 8-byte header
            header = f.read(8)
            if len(header) < 8:
                break  # End of file
            
            # Extract pl_type (record type) from header
            # Header format: (dummy_int: 4 bytes, pl_type: 4 bytes)
            try:
                _, pl_type = struct.unpack('<ii', header)
            except struct.error as exc:
                raise MapParseError(f"Corrupt header near byte {f.tell()}") from exc
            
            # Parse record based on type
            if pl_type == 0:  # Snapshot record
                raw = f.read(78)  # Snapshot body is 78 bytes
                record, snap_offset = _read_snap_record(raw, idx_snap + 1, snap_offset)
                wave_map.snapshots.append(record)
                idx_snap += 1
                # Skip padding: 128 total - 8 header - 78 body = 42 bytes
                f.seek(42, os.SEEK_CUR)
                
            elif pl_type == 1:  # History record
                raw = f.read(65)  # History body is 65 bytes
                record, hist_offset = _read_hist_record(raw, idx_hist + 1, hist_offset)
                wave_map.histories.append(record)
                idx_hist += 1
                # Skip padding: 128 - 8 - 65 = 55 bytes
                f.seek(55, os.SEEK_CUR)
                
            elif pl_type == 2:  # Dump record
                raw = f.read(78)  # Dump body is 78 bytes
                record, dump_offset = _read_dump_record(raw, idx_dump + 1, dump_offset)
                wave_map.dumps.append(record)
                idx_dump += 1
                # Skip padding: 128 - 8 - 78 = 42 bytes
                f.seek(42, os.SEEK_CUR)
                
            elif pl_type == 3:  # Geometry record
                raw = f.read(76)  # Geometry body is 76 bytes
                record, geom_offset = _read_geom_record(raw, idx_geom + 1, geom_offset)
                wave_map.geometries.append(record)
                idx_geom += 1
                # Skip padding: 128 - 8 - 76 = 44 bytes
                f.seek(44, os.SEEK_CUR)
                
            elif pl_type == 4:  # Crack record
                raw = f.read(36)  # Crack body is 36 bytes
                record, crack_offset = _read_crack_record(raw, idx_crack + 1, crack_offset)
                wave_map.crack_data.append(record)
                idx_crack += 1
                # Skip padding: 128 - 8 - 36 = 84 bytes
                f.seek(84, os.SEEK_CUR)
                
            else:  # Unknown record type
                raise MapParseError(f"Unknown record type {pl_type} at byte {f.tell()-8}")
    
    return wave_map
