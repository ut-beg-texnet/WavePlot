"""Utility classes for reading geometry data from the legacy .geo files."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from .data_loader import DataFileMissing
from .map_parser import GeomMapRecord, WaveMap


STATE_ID = 0x01FEDCB1
BOUND_ID = 0x01FEDCB2
BLOCK_SIZE = 128


@dataclass(frozen=True)
class PropertyRecord:
    bulk: float
    shear: float
    density: float


@dataclass(frozen=True)
class MaterialRecord:
    mat_num: int
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int


@dataclass(frozen=True)
class SourceRecord:
    stype: int
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int


@dataclass(frozen=True)
class StopeRecord:
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
    record: GeomMapRecord
    properties: List[PropertyRecord]
    materials: List[MaterialRecord]
    sources: List[SourceRecord]
    stopes: List[StopeRecord]


class GeometryLoader:
    """Load and cache geometry information from a WAVE project."""

    def __init__(self, wave_map: WaveMap) -> None:
        self.wave_map = wave_map
        self.base_path = os.path.splitext(self.wave_map.map_path)[0]

    def _path(self) -> str:
        path = f"{self.base_path}.geo"
        if not os.path.exists(path):
            raise DataFileMissing(path)
        return path

    def load_geometry(self, record: GeomMapRecord) -> GeometryData:
        index = max(0, record.index - 1)
        return self._load_geometry_cached(index)

    @lru_cache(maxsize=8)
    def _load_geometry_cached(self, index: int) -> GeometryData:
        record = self.wave_map.geometries[index]
        path = self._path()
        props: List[PropertyRecord] = []
        mats: List[MaterialRecord] = []
        sources: List[SourceRecord] = []
        stopes: List[StopeRecord] = []

        with open(path, "rb") as handle:
            pos = record.offset + 4  # skip header length marker

            # Skip state/boundary tables that precede the geometry data blocks.
            while True:
                handle.seek(pos)
                ident_bytes = handle.read(4)
                if len(ident_bytes) < 4:
                    break
                ident = struct.unpack("<i", ident_bytes)[0]
                if ident not in (STATE_ID, BOUND_ID):
                    break
                pos += BLOCK_SIZE

            # Properties --------------------------------------------------
            prop_count = max(0, record.prop_total)
            for _ in range(prop_count):
                handle.seek(pos)
                bulk, shear, density = struct.unpack("<fff", handle.read(12))
                props.append(PropertyRecord(bulk=bulk, shear=shear, density=density))
                pos += BLOCK_SIZE

            # Materials ---------------------------------------------------
            mat_count = max(0, record.mat_total)
            for _ in range(mat_count):
                handle.seek(pos)
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

            # Sources -----------------------------------------------------
            src_count = max(0, record.source_total)
            for _ in range(src_count):
                handle.seek(pos)
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

            # Stopes ------------------------------------------------------
            stope_count = max(0, record.stope_total)
            for _ in range(stope_count):
                handle.seek(pos)
                stope_id, stype, i1, i2, j1, j2, k1, k2 = struct.unpack("<iiiiiiii", handle.read(32))
                handle.seek(pos + 116)
                layout_size = struct.unpack("<i", handle.read(4))[0]
                pos += BLOCK_SIZE
                layout_data: Optional[bytes] = None
                if layout_size > 0:
                    handle.seek(pos)
                    layout_data = handle.read(layout_size)
                    pos += layout_size
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
    material_rects: List[tuple[float, float, float, float, int]]
    stope_rects: List[tuple[float, float, float, float, int]]
    source_points: List[tuple[float, float, int]]

    @staticmethod
    def empty() -> "GeometryOverlay":
        return GeometryOverlay(material_rects=[], stope_rects=[], source_points=[])
