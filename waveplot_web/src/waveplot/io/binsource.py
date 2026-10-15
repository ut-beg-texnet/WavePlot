"""Lazy float32 access to WAVE binary data files (``.snp``, ``.hst``, ``.dmp``).

Every WAVE record is wrapped in Fortran-style int32 length markers. Readers return float32
arrays with the markers stripped. Only the requested plane / series is read from disk.
On Windows files are *not* kept memory-mapped by default (an open mapping locks the file and
would stop a WAVE run from rewriting it); on other platforms a read-only memmap is reused.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

from .records import DumpRecord, HistRecord, SnapRecord


class DataUnavailable(RuntimeError):
    """Requested data lies beyond the end of the file (missing, empty or truncated)."""


class BinSource:
    def __init__(self, path: str | os.PathLike[str] | None, keep_open: bool | None = None):
        self.path = Path(path) if path else None
        self.keep_open = (sys.platform != "win32") if keep_open is None else keep_open
        self._mm: np.memmap | None = None
        self._lock = threading.Lock()

    @property
    def exists(self) -> bool:
        return self.path is not None and self.path.is_file()

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.exists else 0

    def close(self) -> None:
        with self._lock:
            if self._mm is not None:
                mm = self._mm
                self._mm = None
                del mm

    def _memmap(self) -> np.memmap:
        with self._lock:
            if self._mm is None:
                self._mm = np.memmap(self.path, dtype="<f4", mode="r")
            return self._mm

    def read_f4(self, offset: int, count: int) -> np.ndarray:
        """Read ``count`` float32 values starting at float index ``offset``."""
        if not self.exists:
            raise DataUnavailable(f"data file missing: {self.path}")
        if offset < 0 or count < 0 or 4 * (offset + count) > self.size:
            raise DataUnavailable(
                f"{self.path.name}: values {offset}..{offset + count} beyond end of file ({self.size} bytes)")
        if self.keep_open:
            return np.asarray(self._memmap()[offset:offset + count])
        return np.fromfile(self.path, dtype="<f4", count=count, offset=4 * offset)

    def read_i4(self, offset: int, count: int) -> np.ndarray:
        return self.read_f4(offset, count).view("<i4")


def read_snapshot(src: BinSource, rec: SnapRecord, check_markers: bool = False) -> np.ndarray:
    """Snapshot plane as ``(xqty, yqty)`` float32: rows along ``ax1``, columns along ``ax2``."""
    block = src.read_f4(rec.offset, rec.nfloats).reshape(rec.xqty, rec.yqty + 2)
    if check_markers:
        m = block[:, [0, -1]].view("<i4")
        if not np.all(m == 4 * rec.yqty):
            raise DataUnavailable(f"snapshot {rec.number}: bad record markers (offset mismatch?)")
    return np.ascontiguousarray(block[:, 1:-1])


def read_snapshot_rows(src: BinSource, rec: SnapRecord, row0: int, row1: int) -> np.ndarray:
    """Rows ``row0:row1`` (along ``ax1``) of a snapshot without reading the whole plane."""
    row0, row1 = max(0, row0), min(rec.xqty, row1)
    w = rec.yqty + 2
    block = src.read_f4(rec.offset + row0 * w, (row1 - row0) * w).reshape(row1 - row0, w)
    return np.ascontiguousarray(block[:, 1:-1])


def read_history(src: BinSource, rec: HistRecord) -> np.ndarray:
    """History samples (float32, length ``xqty``); each sample is its own Fortran record."""
    block = src.read_f4(rec.offset, rec.nfloats).reshape(rec.xqty, 3)
    return np.ascontiguousarray(block[:, 1])


def read_dump(src: BinSource, rec: DumpRecord) -> np.ndarray:
    """Dump plane as ``(xqty, yqty, vqty)`` float32 (one Fortran record; data at offset + 1)."""
    n = rec.iqty * rec.jqty * rec.kqty * rec.vqty
    data = src.read_f4(rec.offset + 1, n)
    return data.reshape(rec.xqty, rec.yqty, rec.vqty)


def snapshot_marker_ok(src: BinSource, rec: SnapRecord) -> bool:
    """Cheap check of the first and last row markers of a snapshot."""
    w = rec.yqty + 2
    first = src.read_i4(rec.offset, 1)[0]
    last = src.read_i4(rec.offset + rec.nfloats - 1, 1)[0]
    return int(first) == 4 * rec.yqty and int(last) == 4 * rec.yqty and w > 2
