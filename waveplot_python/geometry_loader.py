"""Utility classes for reading geometry data from the legacy .geo files.

This module provides classes for loading and processing geometry information from
WAVE .geo files, including material properties, material regions, sources, and stopes.
The geometry data can be overlaid on snapshots and dumps for visualization.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from .data_loader import DataFileMissing
from .map_parser import GeomMapRecord, WaveMap

# Magic identifiers for state and boundary tables in .geo files
STATE_ID = 0x01FEDCB1
BOUND_ID = 0x01FEDCB2
# Fixed block size for geometry records (matches Turbo Pascal format)
BLOCK_SIZE = 128


@dataclass(frozen=True)
class PropertyRecord:
    """Material property record containing bulk modulus, shear modulus, and density.
    
    Attributes:
        bulk: Bulk modulus value
        shear: Shear modulus value
        density: Material density value
    """
    bulk: float
    shear: float
    density: float


@dataclass(frozen=True)
class MaterialRecord:
    """Material region record defining a rectangular block of material.
    
    Attributes:
        mat_num: Material number identifier
        i1, i2: i-axis range (inclusive)
        j1, j2: j-axis range (inclusive)
        k1, k2: k-axis range (inclusive)
    """
    mat_num: int
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int


@dataclass(frozen=True)
class SourceRecord:
    """Source region record defining a rectangular source area.
    
    Attributes:
        stype: Source type identifier
        i1, i2: i-axis range (inclusive)
        j1, j2: j-axis range (inclusive)
        k1, k2: k-axis range (inclusive)
    """
    stype: int
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int


@dataclass(frozen=True)
class StopeRecord:
    """Stope region record defining a mined-out area.
    
    Attributes:
        stope_id: Unique stope identifier
        stype: Stope type identifier
        i1, i2: i-axis range (inclusive)
        j1, j2: j-axis range (inclusive)
        k1, k2: k-axis range (inclusive)
        layout_size: Size of layout data in bytes (0 if no layout)
        layout: Optional binary layout data
    """
    stope_id: int
    stype: int
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    layout_size: int
    layout: Optional[bytes]


@dataclass(frozen=True)
class GeometryData:
    """Complete geometry data for a WAVE project.
    
    Contains all geometry information parsed from a .geo file, including
    material properties, material regions, sources, and stopes.
    
    Attributes:
        record: The geometry map record containing metadata
        properties: List of material property records
        materials: List of material region records
        sources: List of source region records
        stopes: List of stope region records
    """
    record: GeomMapRecord
    properties: List[PropertyRecord]
    materials: List[MaterialRecord]
    sources: List[SourceRecord]
    stopes: List[StopeRecord]


class GeometryLoader:
    """Load and cache geometry information from a WAVE project.
    
    This class reads geometry data from .geo files based on offset information
    in the parsed map file. Results are cached to avoid reloading the same geometry.
    """

    def __init__(self, wave_map: WaveMap) -> None:
        """Initialize the geometry loader.
        
        Args:
            wave_map: Parsed WaveMap containing geometry record metadata
        """
        self.wave_map = wave_map
        # Extract base path to construct .geo file path
        self.base_path = os.path.splitext(self.wave_map.map_path)[0]

    def _path(self) -> str:
        """Construct path to the .geo file and verify it exists.
        
        Returns:
            Full path to the geometry file
            
        Raises:
            DataFileMissing: If the .geo file does not exist
        """
        path = f"{self.base_path}.geo"
        if not os.path.exists(path):
            raise DataFileMissing(path)
        return path

    def load_geometry(self, record: GeomMapRecord) -> GeometryData:
        """Load geometry data for a given geometry record.
        
        Args:
            record: Geometry map record containing offset and metadata
            
        Returns:
            GeometryData object containing all geometry information
        """
        # Convert 1-based index to 0-based array index
        index = max(0, record.index - 1)
        return self._load_geometry_cached(index)

    @lru_cache(maxsize=8)
    def _load_geometry_cached(self, index: int) -> GeometryData:
        """Load geometry data from file (cached for performance).
        
        This method reads the binary .geo file structure:
        1. Skips state/boundary identifier blocks
        2. Reads property records (bulk, shear, density)
        3. Reads material region records
        4. Reads source region records
        5. Reads stope records (with optional layout data)
        
        Args:
            index: 0-based index into the geometries list
            
        Returns:
            GeometryData object containing all parsed geometry information
        """
        record = self.wave_map.geometries[index]
        path = self._path()
        props: List[PropertyRecord] = []
        mats: List[MaterialRecord] = []
        sources: List[SourceRecord] = []
        stopes: List[StopeRecord] = []

        with open(path, "rb") as handle:
            # Start after the 4-byte header length marker
            pos = record.offset + 4

            # Skip state/boundary identifier blocks that precede geometry data
            # These blocks are identified by magic numbers STATE_ID or BOUND_ID
            while True:
                handle.seek(pos)
                ident_bytes = handle.read(4)
                if len(ident_bytes) < 4:
                    break
                ident = struct.unpack("<i", ident_bytes)[0]
                if ident not in (STATE_ID, BOUND_ID):
                    break
                pos += BLOCK_SIZE

            # Read property records (bulk modulus, shear modulus, density)
            prop_count = max(0, record.prop_total)
            for _ in range(prop_count):
                handle.seek(pos)
                # Each property record: 3 floats (12 bytes) + padding to BLOCK_SIZE
                bulk, shear, density = struct.unpack("<fff", handle.read(12))
                props.append(PropertyRecord(bulk=bulk, shear=shear, density=density))
                pos += BLOCK_SIZE

            # Read material region records
            mat_count = max(0, record.mat_total)
            for _ in range(mat_count):
                handle.seek(pos)
                # Each material record: mat_num + 6 integers (i1,i2,j1,j2,k1,k2) = 28 bytes
                mat_num, i1, i2, j1, j2, k1, k2 = struct.unpack("<iiiiiii", handle.read(28))
                mats.append(
                    MaterialRecord(
                        mat_num=mat_num,
                        i1=i1,
                        i2=i2,
                        j1=j1,
                        j2=j2,
                        k1=k1,
                        k2=k2,
                    )
                )
                pos += BLOCK_SIZE

            # Read source region records
            src_count = max(0, record.source_total)
            for _ in range(src_count):
                handle.seek(pos)
                # Each source record: stype + 6 integers (i1,i2,j1,j2,k1,k2) = 28 bytes
                stype, i1, i2, j1, j2, k1, k2 = struct.unpack("<iiiiiii", handle.read(28))
                sources.append(
                    SourceRecord(
                        stype=stype,
                        i1=i1,
                        i2=i2,
                        j1=j1,
                        j2=j2,
                        k1=k1,
                        k2=k2,
                    )
                )
                pos += BLOCK_SIZE

            # Read stope records (mined-out regions)
            stope_count = max(0, record.stope_total)
            for _ in range(stope_count):
                handle.seek(pos)
                # Stope header: stope_id + stype + 6 integers = 32 bytes
                stope_id, stype, i1, i2, j1, j2, k1, k2 = struct.unpack("<iiiiiiii", handle.read(32))
                # Layout size is stored at offset 116 within the block
                handle.seek(pos + 116)
                layout_size = struct.unpack("<i", handle.read(4))[0]
                pos += BLOCK_SIZE
                # Read optional layout data if present
                layout_data: Optional[bytes] = None
                if layout_size > 0:
                    handle.seek(pos)
                    layout_data = handle.read(layout_size)
                    pos += layout_size
                    # Align to next BLOCK_SIZE boundary
                    padding = (-layout_size) % BLOCK_SIZE
                    if padding:
                        pos += padding
                stopes.append(
                    StopeRecord(
                        stope_id=stope_id,
                        stype=stype,
                        i1=i1,
                        i2=i2,
                        j1=j1,
                        j2=j2,
                        k1=k1,
                        k2=k2,
                        layout_size=layout_size,
                        layout=layout_data,
                    )
                )

        return GeometryData(
            record=record,
            properties=props,
            materials=mats,
            sources=sources,
            stopes=stopes,
        )


@dataclass(frozen=True)
class GeometryOverlay:
    """2D overlay representation of geometry for plotting on snapshots/dumps.
    
    This class provides a flattened 2D view of 3D geometry data, suitable for
    overlaying on 2D plots. Rectangles are represented as (x0, y0, width, height, id)
    tuples, and source points as (x, y, type) tuples.
    
    Attributes:
        material_rects: List of (x0, y0, width, height, mat_num) tuples for material regions
        stope_rects: List of (x0, y0, width, height, stope_id) tuples for stope regions
        source_points: List of (x, y, stype) tuples for source locations
    """
    material_rects: List[tuple[float, float, float, float, int]]
    stope_rects: List[tuple[float, float, float, float, int]]
    source_points: List[tuple[float, float, int]]

    @staticmethod
    def empty() -> "GeometryOverlay":
        """Create an empty geometry overlay (no geometry data).
        
        Returns:
            GeometryOverlay with empty lists for all geometry types
        """
        return GeometryOverlay(material_rects=[], stope_rects=[], source_points=[])
