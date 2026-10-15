"""Grid-index ↔ physical-coordinate conventions.

WAVE uses 1-based grid indices. Cell (zone) ``i`` spans ``[(i-1)·dx, i·dx]`` in metres
(v407 maps index ``X`` to ``(X-1)·dx``, DSPL.PAS:635-639). Values are placed at cell centres,
shifted by half a cell along axes where the variable is staggered (``CalcPix``,
DSPSNAP.PAS:5-28). In plots the first record axis (``ax1``) is horizontal and the second
(``ax2``) vertical; a vertical ``j`` axis points up, other vertical axes point down
(DSPL.PAS:728-729).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.records import DumpRecord, Extent, HistRecord, SnapRecord

AXIS_LABEL = {"i": "x", "j": "y", "k": "z", "t": "t"}


@dataclass(frozen=True)
class AxisCoords:
    axis: str                   # 'i' | 'j' | 'k' | 't'
    index: np.ndarray           # grid index (int) or cycle-like index for 't'
    coord: np.ndarray           # cell-centre coordinate (m) or time (s)
    spacing: float              # cell size along the axis (m) or sample interval (s)
    step: int                   # sampling stride in cells

    @property
    def label(self) -> str:
        return AXIS_LABEL.get(self.axis, self.axis)

    @property
    def unit(self) -> str:
        return "s" if self.axis == "t" else "m"

    @property
    def edges(self) -> tuple[float, float]:
        """Physical extent covered by the samples (outer cell edges)."""
        if len(self.coord) == 0:
            return (0.0, 0.0)
        half = 0.5 * self.spacing * (self.step if self.axis != "t" else 1)
        return float(self.coord[0] - half), float(self.coord[-1] + half)

    def nearest(self, value: float) -> int:
        """Array position nearest to a physical coordinate."""
        return int(np.clip(np.abs(self.coord - value).argmin(), 0, len(self.coord) - 1))


def _first_index(lo: int, hi: int, step: int) -> int:
    step = max(step, 1)
    return lo + ((hi - lo) % step) // 2


def spatial_axis(axis: str, extent: Extent, n: int, spacing: float, step: int, shift: int) -> AxisCoords:
    lo, hi = extent.range(axis)
    step = max(int(step), 1)
    idx = _first_index(lo, hi, step) + step * np.arange(n)
    coord = (idx - 0.5 + 0.5 * shift) * spacing
    return AxisCoords(axis, idx.astype(np.int64), coord.astype(np.float64), float(spacing), step)


def time_axis(t1: float, t2: float, n: int, model_dt: float, step: int) -> AxisCoords:
    coord = np.linspace(t1, t2, n) if n > 1 else np.array([t1], dtype=float)
    dt = (t2 - t1) / (n - 1) if n > 1 else float(model_dt)
    idx = np.rint(coord / model_dt).astype(np.int64) if model_dt else np.arange(n)
    return AxisCoords("t", idx, coord, float(dt), max(int(step), 1))


def snapshot_axes(rec: SnapRecord) -> tuple[AxisCoords, AxisCoords]:
    """Coordinates of a snapshot's rows (``ax1``) and columns (``ax2``)."""
    s1, s2 = rec.stagger
    if rec.ax1 == "t":
        a1 = time_axis(rec.t1, rec.time, rec.xqty, rec.dx, rec.sx)
    else:
        a1 = spatial_axis(rec.ax1, rec.extent, rec.xqty, rec.dx, rec.sx, s1)
    if rec.ax2 == "t":
        a2 = time_axis(rec.t1, rec.time, rec.yqty, rec.dy, rec.sy)
    else:
        a2 = spatial_axis(rec.ax2, rec.extent, rec.yqty, rec.dy, rec.sy, s2)
    return a1, a2


def dump_axes(rec: DumpRecord) -> tuple[AxisCoords, AxisCoords]:
    d1, d2 = rec.spacing
    st1, st2 = rec.steps
    return (spatial_axis(rec.ax1, rec.extent, rec.xqty, d1, st1, 0),
            spatial_axis(rec.ax2, rec.extent, rec.yqty, d2, st2, 0))


def history_axis(rec: HistRecord) -> np.ndarray:
    return rec.tstart + rec.dt * np.arange(rec.n)


def vertical_up(ax2: str) -> bool:
    """Whether the vertical plot axis (``ax2``) increases upwards (only ``j``, as in v407)."""
    return ax2 == "j"


def cell_center(index: float, spacing: float, shift: int = 0) -> float:
    return (index - 0.5 + 0.5 * shift) * spacing


def cell_edges(lo: int, hi: int, spacing: float) -> tuple[float, float]:
    """Physical edges of the cell range ``lo..hi`` (inclusive)."""
    return (lo - 1) * spacing, hi * spacing
