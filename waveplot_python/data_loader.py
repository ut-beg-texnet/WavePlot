"""Load binary WAVE datasets using the parsed .map metadata."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np

from .map_parser import (
    DumpMapRecord,
    HistMapRecord,
    SnapMapRecord,
    WaveMap,
)

_FLOAT32 = np.dtype('<f4')


class DataFileMissing(FileNotFoundError):
    """we raise this when a required binary file is missing"""


class WaveDataLoader:
    """Provides numpy-friendly access to the WAVE binary blobs."""

    def __init__(self, wave_map: WaveMap) -> None:
        self.wave_map = wave_map
        self.base_path = os.path.splitext(self.wave_map.map_path)[0]

    def _path(self, extension: str) -> str:
        path = f"{self.base_path}.{extension}"
        if not os.path.exists(path):
            raise DataFileMissing(path)
        return path

    @staticmethod
    def _read_float_block(path: str, offset_floats: int, count: int) -> np.ndarray:
        # reads a block of float32 data from a file
        with open(path, 'rb') as handle:
            handle.seek(offset_floats * _FLOAT32.itemsize)
            raw = handle.read(count * _FLOAT32.itemsize)
        if len(raw) != count * _FLOAT32.itemsize:
            raise EOFError(f"Unexpected EOF while reading {path}")
        return np.frombuffer(raw, dtype=_FLOAT32).copy()

    def load_snapshot(self, record: SnapMapRecord, *, max_points: Optional[int] = None) -> np.ndarray:
        """Return snapshot as a 2D float32 array (Y x X)."""
        
        path = self._path('snp')
        rows = record.x_qty
        cols = record.y_qty + 2
        block = self._read_float_block(path, record.offset, rows * cols)
        data = block.reshape((rows, cols))[:, 1:-1]
        if max_points:
            eff_rows, eff_cols = data.shape
            if eff_rows > max_points or eff_cols > max_points:
                step_x = max(1, eff_rows // max_points)
                step_y = max(1, eff_cols // max_points)
                data = data[::step_x, ::step_y]
        return data.astype(np.float32)

    def load_history(self, record: HistMapRecord) -> np.ndarray:
        """Return history as 1D float32 array with length == sample_qty."""
        path = self._path('hst')
        count = record.sample_qty * 3
        block = self._read_float_block(path, record.offset, count)
        data = block.reshape((-1, 3))[:, 1]
        return data.astype(np.float32)

    def load_dump(self, record: DumpMapRecord) -> np.ndarray:
        """Return dump values as ndarray of shape (k, i, j, v)."""
        path = self._path('dmp')
        k_qty = max(record.k_qty, 1)
        cell_count = record.i_qty * record.j_qty * k_qty
        total = cell_count * record.v_qty + 2
        block = self._read_float_block(path, record.offset, total)
        core = block[1:-1]  # strip sentinels
        data = core.reshape((k_qty, record.i_qty, record.j_qty, record.v_qty))
        return data.astype(np.float32)

    @lru_cache(maxsize=8)
    def snapshot_series(self, indices: Iterable[int], *, max_points: Optional[int] = None) -> list[np.ndarray]:
        """Return list of snapshot arrays for convenience (cached)."""
        result = []
        for idx in indices:
            record = self.wave_map.snapshots[idx]
            result.append(self.load_snapshot(record, max_points=max_points))
        return result

    def time_axis(self, record: HistMapRecord) -> np.ndarray:
        # returns the time axis for a history record
        # count is the number of samples
        # start is the start time
        # dt is the time step
        # returns the time axis as a numpy array
        count = record.sample_qty
        start = record.t_start
        dt = record.dt if record.dt else (record.t_end - record.t_start) / max(count - 1, 1)
        return (start + dt * np.arange(count, dtype=np.float32)).astype(np.float32)

    def available_files(self) -> dict[str, bool]:
        # function to check if the binary files are present
        return {ext: os.path.exists(f"{self.base_path}.{ext}") for ext in ['snp', 'hst', 'dmp', 'geo', 'crk']}
