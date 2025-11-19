"""Load binary WAVE datasets using the parsed .map metadata.

This module provides the WaveDataLoader class which reads binary data files (.snp, .hst, .dmp)
based on offset information parsed from the .map file. The loader handles reading
float32 arrays from the binary files at the correct file positions.
"""

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

# Little-endian float32 data type matching the Turbo Pascal format
_FLOAT32 = np.dtype('<f4')


class DataFileMissing(FileNotFoundError):
    """Exception raised when a required binary data file (.snp, .hst, .dmp, etc.) is missing."""


class WaveDataLoader:
    """Provides numpy-friendly access to the WAVE binary data files.
    
    This class reads binary data from .snp (snapshots), .hst (histories), and .dmp (dumps)
    files based on offset information stored in the parsed WaveMap. All data is read
    as little-endian float32 arrays matching the Turbo Pascal format.
    """

    def __init__(self, wave_map: WaveMap) -> None:
        """Initialize the data loader with a parsed map file.
        
        Args:
            wave_map: Parsed WaveMap containing record metadata and file offsets
        """
        self.wave_map = wave_map
        # Extract base path (without .map extension) to construct data file paths
        self.base_path = os.path.splitext(self.wave_map.map_path)[0]

    def _path(self, extension: str) -> str:
        """Construct path to a binary data file and verify it exists.
        
        Args:
            extension: File extension (e.g., 'snp', 'hst', 'dmp')
            
        Returns:
            Full path to the data file
            
        Raises:
            DataFileMissing: If the file does not exist
        """
        path = f"{self.base_path}.{extension}"
        if not os.path.exists(path):
            raise DataFileMissing(path)
        return path

    @staticmethod
    def _read_float_block(path: str, offset_floats: int, count: int) -> np.ndarray:
        """Read a block of float32 values from a binary file.
        
        Args:
            path: Path to the binary file
            offset_floats: Starting position in float32 units (not bytes)
            count: Number of float32 values to read
            
        Returns:
            NumPy array of float32 values
            
        Raises:
            EOFError: If the file is truncated or doesn't contain enough data
        """
        with open(path, 'rb') as handle:
            # Seek to byte position: offset_floats * 4 bytes per float32
            handle.seek(offset_floats * _FLOAT32.itemsize)
            raw = handle.read(count * _FLOAT32.itemsize)
        if len(raw) != count * _FLOAT32.itemsize:
            raise EOFError(f"Unexpected EOF while reading {path}")
        # Convert bytes to numpy array and return a copy (not a view)
        return np.frombuffer(raw, dtype=_FLOAT32).copy()

    def load_snapshot(self, record: SnapMapRecord, *, max_points: Optional[int] = None) -> np.ndarray:
        """Load snapshot data as a 2D float32 array.
        
        Snapshots are stored as (x_qty x (y_qty + 2)) arrays with padding columns.
        The padding is removed, resulting in a (x_qty x y_qty) array.
        
        Args:
            record: Snapshot record containing offset and dimensions
            max_points: Optional maximum points per dimension for downsampling.
                       If provided, data is subsampled to reduce memory usage.
        
        Returns:
            2D numpy array of shape (x_qty, y_qty) with float32 values
        """
        path = self._path('snp')
        rows = record.x_qty
        cols = record.y_qty + 2  # Includes padding columns
        block = self._read_float_block(path, record.offset, rows * cols)
        # Reshape and remove padding columns (first and last)
        data = block.reshape((rows, cols))[:, 1:-1]
        # Optional downsampling for large datasets
        if max_points:
            eff_rows, eff_cols = data.shape
            if eff_rows > max_points or eff_cols > max_points:
                step_x = max(1, eff_rows // max_points)
                step_y = max(1, eff_cols // max_points)
                data = data[::step_x, ::step_y]
        return data.astype(np.float32)

    def load_history(self, record: HistMapRecord) -> np.ndarray:
        """Load history data as a 1D float32 array.
        
        History files store data in groups of 3 floats per sample. This function
        extracts the middle value (index 1) from each group.
        
        Args:
            record: History record containing offset and sample count
            
        Returns:
            1D numpy array of length sample_qty with float32 values
        """
        path = self._path('hst')
        # Each sample is stored as 3 floats, we extract the middle one
        count = record.sample_qty * 3
        block = self._read_float_block(path, record.offset, count)
        # Reshape to (samples, 3) and extract column 1 (the actual values)
        data = block.reshape((-1, 3))[:, 1]
        return data.astype(np.float32)

    def load_dump(self, record: DumpMapRecord) -> np.ndarray:
        """Load dump volume data as a 4D float32 array.
        
        Dump files contain 3D grid data with multiple variables. The data is stored
        with sentinel values at the start and end that are removed.
        
        Args:
            record: Dump record containing offset and grid dimensions
            
        Returns:
            4D numpy array of shape (k_qty, i_qty, j_qty, v_qty) with float32 values
        """
        path = self._path('dmp')
        k_qty = max(record.k_qty, 1)  # Ensure at least 1 layer
        cell_count = record.i_qty * record.j_qty * k_qty
        # Total includes 2 sentinel values (one at start, one at end)
        total = cell_count * record.v_qty + 2
        block = self._read_float_block(path, record.offset, total)
        core = block[1:-1]  # Remove sentinel values
        # Reshape to (k, i, j, variables)
        data = core.reshape((k_qty, record.i_qty, record.j_qty, record.v_qty))
        return data.astype(np.float32)

    @lru_cache(maxsize=8)
    def snapshot_series(self, indices: Iterable[int], *, max_points: Optional[int] = None) -> list[np.ndarray]:
        """Load multiple snapshots and return as a list (cached for performance).
        
        This convenience method loads multiple snapshots at once. Results are cached
        to avoid reloading the same data.
        
        Args:
            indices: Iterable of snapshot indices to load
            max_points: Optional maximum points per dimension for downsampling
            
        Returns:
            List of 2D numpy arrays, one per snapshot index
        """
        result = []
        for idx in indices:
            record = self.wave_map.snapshots[idx]
            result.append(self.load_snapshot(record, max_points=max_points))
        return result

    def time_axis(self, record: HistMapRecord) -> np.ndarray:
        """Generate the time axis array for a history record.
        
        Creates a uniformly spaced time array from t_start to t_end based on
        the sample count and time step (dt). If dt is not available, it's computed
        from the time span and sample count.
        
        Args:
            record: History record containing time range and sample information
            
        Returns:
            1D numpy array of time values in seconds (float32)
        """
        count = record.sample_qty
        start = record.t_start
        # Compute dt if not available (e.g., for single-sample histories)
        dt = record.dt if record.dt else (record.t_end - record.t_start) / max(count - 1, 1)
        return (start + dt * np.arange(count, dtype=np.float32)).astype(np.float32)

    def available_files(self) -> dict[str, bool]:
        """Check which binary data files are present in the project directory.
        
        Returns:
            Dictionary mapping file extensions ('snp', 'hst', 'dmp', 'geo', 'crk')
            to boolean values indicating file existence
        """
        return {ext: os.path.exists(f"{self.base_path}.{ext}") for ext in ['snp', 'hst', 'dmp', 'geo', 'crk']}
