"""Server-side state shared by callbacks: configuration, dataset registry, small helpers."""

from __future__ import annotations

import logging
from typing import Any

from ..config import AppConfig
from ..model.dataset import WaveDataset
from ..model.registry import DatasetRegistry

log = logging.getLogger("waveplot.web")

_CONFIG: AppConfig | None = None
_REGISTRY: DatasetRegistry | None = None


def init(cfg: AppConfig) -> DatasetRegistry:
    global _CONFIG, _REGISTRY
    _CONFIG = cfg
    _REGISTRY = DatasetRegistry(cfg.data_roots, max_open=cfg.max_open_datasets, cache_bytes=cfg.cache_mb * 2**20)
    return _REGISTRY


def config() -> AppConfig:
    if _CONFIG is None:
        raise RuntimeError("web state not initialised")
    return _CONFIG


def registry() -> DatasetRegistry:
    if _REGISTRY is None:
        raise RuntimeError("web state not initialised")
    return _REGISTRY


def get_dataset(ds_id: str | None) -> WaveDataset | None:
    if not ds_id:
        return None
    try:
        return registry().get(ds_id)
    except (KeyError, PermissionError, FileNotFoundError, OSError, ValueError) as exc:
        log.warning("cannot open dataset %s: %s", ds_id, exc)
        return None


def prefs_value(prefs: dict[str, Any] | None, key: str, default: Any) -> Any:
    return (prefs or {}).get(key, default)


def mode_of(prefs: dict[str, Any] | None) -> str:
    m = prefs_value(prefs, "mode", "standard")
    return m if m in ("standard", "v407") else "standard"


def scheme_of(prefs: dict[str, Any] | None) -> str:
    s = prefs_value(prefs, "scheme", "light")
    return s if s in ("light", "dark") else "light"
