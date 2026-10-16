"""``Trace``: a 1-D signal with its own axis (the v407 ``Hmap`` + ``GRs`` pair)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    y: np.ndarray
    t0: float = 0.0
    dt: float = 1.0
    x_unit: str = "s"          # "s" | "m" | "Hz"
    y_unit: str = ""
    label: str = ""
    scale_hint: str | None = None   # "log" for amplitude spectra (v407 switched to log display)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def x(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(len(self.y))

    @property
    def t_end(self) -> float:
        return self.t0 + self.dt * (len(self.y) - 1)

    def replace(self, **changes) -> Trace:
        return dataclasses.replace(self, **changes)

    def with_y(self, y: np.ndarray, **changes) -> Trace:
        return dataclasses.replace(self, y=np.asarray(y, dtype=float), **changes)
