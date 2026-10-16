"""Process-wide registry of datasets under configured data roots, plus a byte-bounded array cache.

Datasets are addressed by ids ``"<root name>/<relative/path/stem>"`` — never by absolute path —
and every resolved path is checked to lie inside its root.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cachetools import LRUCache

from ..io.discover import find_maps
from .dataset import WaveDataset, open_dataset


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    root: str
    relative: str
    path: Path
    size: int
    mtime: float

    @property
    def name(self) -> str:
        return self.path.stem


def _nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return max(int(value.nbytes), 1)
    if isinstance(value, tuple | list):
        return sum(_nbytes(v) for v in value) or 64
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values()) or 64
    return 256


class ByteLRU:
    """Thread-safe LRU cache bounded by the total size of cached NumPy arrays."""

    def __init__(self, max_bytes: int = 512 * 2**20):
        self._cache: LRUCache = LRUCache(maxsize=max_bytes, getsizeof=_nbytes)
        self._lock = threading.Lock()

    def get_or_compute(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        value = fn()
        with self._lock:
            try:
                self._cache[key] = value
            except ValueError:  # larger than the whole cache
                pass
        return value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def currsize(self) -> int:
        return int(self._cache.currsize)


class DatasetRegistry:
    def __init__(self, roots: dict[str, Path], max_open: int = 8, cache_bytes: int = 512 * 2**20):
        self.roots = {name: Path(p).resolve() for name, p in roots.items()}
        self.max_open = max_open
        self.cache = ByteLRU(cache_bytes)
        self._open: OrderedDict[tuple, WaveDataset] = OrderedDict()
        self._lock = threading.RLock()
        self._listing: list[DatasetEntry] | None = None

    # --- discovery ----------------------------------------------------------------------------
    def list_datasets(self, refresh: bool = False) -> list[DatasetEntry]:
        if self._listing is not None and not refresh:
            return self._listing
        out: list[DatasetEntry] = []
        for name, root in self.roots.items():
            if not root.is_dir():
                continue
            for p in find_maps(root):
                rel = p.relative_to(root).with_suffix("").as_posix()
                st = p.stat()
                out.append(DatasetEntry(f"{name}/{rel}", name, rel, p, st.st_size, st.st_mtime))
        self._listing = out
        return out

    def resolve(self, dataset_id: str) -> Path:
        """Map a dataset id to its ``.map`` path, refusing anything outside the data roots."""
        if not dataset_id or "\x00" in dataset_id:
            raise KeyError(dataset_id)
        root_name, _, rel = dataset_id.partition("/")
        root = self.roots.get(root_name)
        if root is None or not rel:
            raise KeyError(dataset_id)
        candidate = (root / rel).resolve()
        if not candidate.is_relative_to(root):
            raise PermissionError(f"{dataset_id} is outside data root {root_name}")
        for ext in (".map", ".MAP"):
            p = candidate.with_name(candidate.name + ext)
            if p.is_file():
                return p
        for e in self.list_datasets():
            if e.id == dataset_id:
                return e.path
        raise KeyError(dataset_id)

    # --- opening -------------------------------------------------------------------------------
    def get(self, dataset_id: str) -> WaveDataset:
        path = self.resolve(dataset_id)
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        with self._lock:
            ds = self._open.get(key)
            if ds is not None:
                self._open.move_to_end(key)
                return ds
            for k in [k for k in self._open if k[0] == key[0]]:   # stale version of the same file
                self._open.pop(k).close()
            ds = open_dataset(path)
            ds.registry_key = key  # type: ignore[attr-defined]
            self._open[key] = ds
            while len(self._open) > self.max_open:
                _, old = self._open.popitem(last=False)
                old.close()
            return ds

    def key_for(self, ds: WaveDataset) -> tuple:
        return getattr(ds, "registry_key", (str(ds.path),))

    def close_all(self) -> None:
        with self._lock:
            for ds in self._open.values():
                ds.close()
            self._open.clear()
            self.cache.clear()
