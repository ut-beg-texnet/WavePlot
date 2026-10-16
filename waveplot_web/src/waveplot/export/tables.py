"""Tabular exports: CSV and the legacy v407 ``.ASC`` / ``.TBL`` text formats."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def frame_csv(series, n: int) -> str:
    """One snapshot frame as CSV in long format: x, y, value (with grid indices)."""
    a1, a2 = series.coords()
    z = series.frame(n)
    xx, yy = np.meshgrid(a1.coord, a2.coord, indexing="ij")
    ii, jj = np.meshgrid(a1.index, a2.index, indexing="ij")
    buf = io.StringIO()
    r = series.records[n]
    buf.write(f"# {series.long_name} ({series.units}); series {series.slug}; t = {r.time:.8g} s\n")
    buf.write(f"{a1.axis}_index,{a2.axis}_index,{a1.label}_{a1.unit},{a2.label}_{a2.unit},{series.var_name}\n")
    data = np.column_stack([ii.ravel(), jj.ravel(), xx.ravel(), yy.ravel(), z.ravel()])
    np.savetxt(buf, data, delimiter=",", fmt=["%d", "%d", "%.6g", "%.6g", "%.8g"])
    return buf.getvalue()


def traces_csv(traces: Sequence) -> str:
    """Several traces as CSV columns (each trace keeps its own x column)."""
    buf = io.StringIO()
    n = max(len(t.y) for t in traces) if traces else 0
    header = []
    for t in traces:
        header += [f"{t.label} {t.x_unit}", f"{t.label}" + (f" ({t.y_unit})" if t.y_unit else "")]
    buf.write(",".join(h.replace(",", ";") for h in header) + "\n")
    cols = []
    for t in traces:
        x = t.x
        cols.append(np.pad(x.astype(float), (0, n - len(x)), constant_values=np.nan))
        cols.append(np.pad(t.y.astype(float), (0, n - len(t.y)), constant_values=np.nan))
    if cols:
        arr = np.column_stack(cols)
        for row in arr:
            buf.write(",".join("" if np.isnan(v) else f"{v:.10g}" for v in row) + "\n")
    return buf.getvalue()


def write_csv(path: str | Path, traces: Sequence) -> Path:
    path = Path(path)
    path.write_text(traces_csv(traces))
    return path


def legacy_asc(traces: Sequence, with_time: bool = True, mode: str = "standard") -> str:
    """v407 ``.ASC`` export (``DI_CvertAscii``): one line per sample, columns of width 14.

    v407 used trace 1's time base and length for every column; in standard mode each column is
    written over the common sample count with the first trace's time column, padded with blanks.
    """
    if not traces:
        return ""
    t0 = traces[0]
    n = len(t0.y) if mode == "v407" else max(len(t.y) for t in traces)
    lines = []
    for k in range(n):
        parts = []
        if with_time:
            parts.append(f"{t0.t0 + k * t0.dt:14.6E}")
        for t in traces:
            parts.append(f"{t.y[k]:14.6E}" if k < len(t.y) else " " * 14)
        lines.append("".join(parts))
    return "\n".join(lines) + "\n"


def legacy_tbl(trace) -> str:
    """v407 ``.TBL`` export: sample count, then one value per line (WAVE ``wa=tab`` source table)."""
    return f"{len(trace.y)}\n" + "\n".join(f"{v:.7E}" for v in trace.y) + "\n"
