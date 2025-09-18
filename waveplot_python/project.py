"""High level project orchestration for the WavePlot Python port."""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional band-pass support
    from scipy.signal import butter, filtfilt, lfilter  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - SciPy optional
    _SCIPY_AVAILABLE = False

from .data_loader import DataFileMissing, WaveDataLoader
from .map_parser import (
    DumpMapRecord,
    GeomMapRecord,
    HistMapRecord,
    SnapMapRecord,
    WaveMap,
    parse_map,
)
from .geometry_loader import GeometryData, GeometryLoader, GeometryOverlay


@dataclass
class FormulaResult:
    axis: np.ndarray
    values: np.ndarray
    axis_label: str
    description: str


class _SafeEvaluator(ast.NodeVisitor):
    _ALLOWED_NODES = (
        ast.Expression,
        ast.UnaryOp,
        ast.BinOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Tuple,
        ast.List,
    )

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(f"Unsupported expression element: {ast.dump(node)}")
        super().generic_visit(node)


class WaveProject:
    """Loads Turbo Pascal WAVE datasets and exposes plotting-friendly helpers."""

    def __init__(self, map_path: str) -> None:
        self.map_path = os.path.abspath(map_path)
        self.wave_map: WaveMap = parse_map(self.map_path)
        self.loader = WaveDataLoader(self.wave_map)
        self._geometry_loader = GeometryLoader(self.wave_map)
        self._geometry_lookup: Dict[Tuple[int, int], int] = {
            (geom.gnum, geom.gid): idx for idx, geom in enumerate(self.wave_map.geometries)
        }

    # Metadata helpers -----------------------------------------------------

    def snapshot_records(self) -> List[SnapMapRecord]:
        return self.wave_map.snapshots

    def history_records(self) -> List[HistMapRecord]:
        return self.wave_map.histories

    def dump_records(self) -> List[DumpMapRecord]:
        return self.wave_map.dumps

    def files_present(self) -> Dict[str, bool]:
        return self.loader.available_files()

    # Data access ----------------------------------------------------------

    def snapshot_array(self, index: int, *, max_points: Optional[int] = None) -> np.ndarray:
        record = self.wave_map.snapshots[index]
        return self.loader.load_snapshot(record, max_points=max_points)

    def history_series(self, index: int) -> Tuple[np.ndarray, np.ndarray, HistMapRecord]:
        record = self.wave_map.histories[index]
        values = self.loader.load_history(record)
        axis = self.loader.time_axis(record)
        return axis, values, record

    def dump_volume(self, index: int) -> Tuple[np.ndarray, DumpMapRecord]:
        record = self.wave_map.dumps[index]
        volume = self.loader.load_dump(record)
        return volume, record

    def geometry_records(self) -> List[GeomMapRecord]:
        return self.wave_map.geometries

    def snapshot_geometry_overlay(self, index: int) -> GeometryOverlay:
        snap = self.wave_map.snapshots[index]
        geom_rec = self._geometry_record_for_ids(snap.gnum, snap.gid)
        if geom_rec is None:
            return GeometryOverlay.empty()
        try:
            geometry = self._geometry_loader.load_geometry(geom_rec)
        except DataFileMissing:
            return GeometryOverlay.empty()

        plane_axes = (
            self._canonical_axis(snap.ax1),
            self._canonical_axis(snap.ax2),
        )
        axis_ranges = {
            'i': snap.i_range,
            'j': snap.j_range,
            'k': snap.k_range,
        }
        return self._build_geometry_overlay(geometry, plane_axes, axis_ranges)

    def dump_geometry_overlay(self, index: int) -> GeometryOverlay:
        dump = self.wave_map.dumps[index]
        geom_rec = self._geometry_record_for_ids(dump.gnum, dump.gid)
        if geom_rec is None:
            return GeometryOverlay.empty()
        try:
            geometry = self._geometry_loader.load_geometry(geom_rec)
        except DataFileMissing:
            return GeometryOverlay.empty()

        plane_axes = (
            self._canonical_axis(dump.ax1),
            self._canonical_axis(dump.ax2),
        )
        axis_ranges = {
            'i': dump.i_range,
            'j': dump.j_range,
            'k': dump.k_range,
        }
        return self._build_geometry_overlay(geometry, plane_axes, axis_ranges)

    def _geometry_record_for_ids(self, gnum: int, gid: int) -> Optional[GeomMapRecord]:
        idx = self._geometry_lookup.get((gnum, gid))
        if idx is None:
            return None
        if 0 <= idx < len(self.wave_map.geometries):
            return self.wave_map.geometries[idx]
        return None

    @staticmethod
    def _canonical_axis(axis: str) -> str:
        mapping = {'x': 'i', 'y': 'j', 'z': 'k'}
        return mapping.get(axis.lower(), axis.lower())

    def _build_geometry_overlay(
        self,
        geometry: GeometryData,
        plane_axes: Tuple[str, str],
        axis_ranges: Dict[str, Tuple[int, int]],
    ) -> GeometryOverlay:
        plan_a, plan_b = plane_axes
        plan_a = plan_a.lower()
        plan_b = plan_b.lower()
        if plan_a == plan_b:
            return GeometryOverlay.empty()
        if plan_a not in ('i', 'j', 'k') or plan_b not in ('i', 'j', 'k'):
            return GeometryOverlay.empty()

        norm_ranges: Dict[str, Tuple[int, int]] = {}
        for axis in ('i', 'j', 'k'):
            rng = axis_ranges.get(axis, (0, -1))
            lo, hi = int(rng[0]), int(rng[1])
            if lo > hi:
                lo, hi = hi, lo
            norm_ranges[axis] = (lo, hi)

        slice_axis = next((ax for ax in ('i', 'j', 'k') if ax not in (plan_a, plan_b)), None)
        if slice_axis is None:
            return GeometryOverlay.empty()

        slice_limits = norm_ranges[slice_axis]
        offsets = {axis: norm_ranges[axis][0] for axis in ('i', 'j', 'k')}

        def clip_range(span: Tuple[int, int], limits: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            lower = max(span[0], limits[0])
            upper = min(span[1], limits[1])
            if lower > upper:
                return None
            return lower, upper

        material_rects: List[tuple[float, float, float, float, int]] = []
        for mat in geometry.materials:
            spans = {
                'i': (mat.i1, mat.i2),
                'j': (mat.j1, mat.j2),
                'k': (mat.k1, mat.k2),
            }
            slice_span = clip_range(spans[slice_axis], slice_limits)
            if slice_span is None:
                continue
            x_span = clip_range(spans[plan_a], norm_ranges[plan_a])
            y_span = clip_range(spans[plan_b], norm_ranges[plan_b])
            if x_span is None or y_span is None:
                continue
            x0 = float(x_span[0] - offsets[plan_a])
            y0 = float(y_span[0] - offsets[plan_b])
            width = float(x_span[1] - x_span[0] + 1)
            height = float(y_span[1] - y_span[0] + 1)
            material_rects.append((x0, y0, width, height, mat.mat_num))

        stope_rects: List[tuple[float, float, float, float, int]] = []
        for stope in geometry.stopes:
            spans = {
                'i': (stope.i1, stope.i2),
                'j': (stope.j1, stope.j2),
                'k': (stope.k1, stope.k2),
            }
            slice_span = clip_range(spans[slice_axis], slice_limits)
            if slice_span is None:
                continue
            x_span = clip_range(spans[plan_a], norm_ranges[plan_a])
            y_span = clip_range(spans[plan_b], norm_ranges[plan_b])
            if x_span is None or y_span is None:
                continue
            x0 = float(x_span[0] - offsets[plan_a])
            y0 = float(y_span[0] - offsets[plan_b])
            width = float(x_span[1] - x_span[0] + 1)
            height = float(y_span[1] - y_span[0] + 1)
            stope_rects.append((x0, y0, width, height, stope.stope_id))

        source_points: List[tuple[float, float, int]] = []
        for src in geometry.sources:
            spans = {
                'i': (src.i1, src.i2),
                'j': (src.j1, src.j2),
                'k': (src.k1, src.k2),
            }
            if clip_range(spans[slice_axis], slice_limits) is None:
                continue
            x_span = clip_range(spans[plan_a], norm_ranges[plan_a])
            y_span = clip_range(spans[plan_b], norm_ranges[plan_b])
            if x_span is None or y_span is None:
                continue
            x_center = (x_span[0] + x_span[1]) * 0.5 - offsets[plan_a] + 0.5
            y_center = (y_span[0] + y_span[1]) * 0.5 - offsets[plan_b] + 0.5
            source_points.append((float(x_center), float(y_center), src.stype))

        return GeometryOverlay(
            material_rects=material_rects,
            stope_rects=stope_rects,
            source_points=source_points,
        )

    # Formula evaluation ---------------------------------------------------

    def evaluate_history_formula(
        self,
        selection: Sequence[int],
        formula: str,
    ) -> FormulaResult:
        """Evaluate Pascal-style history formula across the selected histories."""
        if not selection:
            raise ValueError("No histories selected for formula evaluation")
        histories = [self.history_series(idx) for idx in selection]
        times_ref = histories[0][0]
        dt_ref = histories[0][2].dt or (times_ref[1] - times_ref[0] if len(times_ref) > 1 else 0.0)
        aligned_values = []
        for axis, values, record in histories:
            if len(axis) != len(times_ref) or not np.allclose(axis, times_ref, atol=1e-6):
                raise ValueError("Selected histories must share the same time axis")
            aligned_values.append(values)
        env: Dict[str, np.ndarray] = {}
        for idx, values in enumerate(aligned_values, start=1):
            env[f"h{idx}"] = values
        python_expr = _convert_formula(formula)
        evaluator = _SafeEvaluator()
        evaluator.visit(ast.parse(python_expr, mode='eval'))
        context = _FormulaContext(dt_ref, times_ref)
        env.update(context.functions())
        env.update({
            'pi': math.pi,
            'e': math.e,
            'np': np,
        })
        result = eval(python_expr, {'__builtins__': {}}, env)  # noqa: P204
        result_array = _ensure_1d(result)
        axis = context.axis_for_result(result_array)
        axis_label = context.axis_label
        desc = formula.strip() or "formula"
        return FormulaResult(axis=axis, values=result_array, axis_label=axis_label, description=desc)


def _convert_formula(formula: str) -> str:
    """Convert Pascal-style tokens [n] to pythonic h{n}."""
    def repl(match: re.Match[str]) -> str:
        idx = match.group(1)
        return f"h{idx}"

    converted = re.sub(r"\[(\d+)\]", repl, formula.strip())
    return converted


class _FormulaContext:
    def __init__(self, dt: float, time_axis: np.ndarray) -> None:
        self.dt = dt
        self.time_axis = time_axis
        self.axis_label = "time (s)"
        self._spectrum_mode = False

    def functions(self) -> Dict[str, object]:
        return {
            'I': self.integrate,
            'D': self.derivative,
            'F': self.fft_magnitude,
            'P': self.fft_phase,
            't': self.band_pass,
            'ABS': np.abs,
            'SQRT': np.sqrt,
            'LOG': np.log,
            'EXP': np.exp,
        }

    def integrate(self, series: np.ndarray) -> np.ndarray:
        return np.cumsum(series) * self.dt

    def derivative(self, series: np.ndarray) -> np.ndarray:
        return np.gradient(series, self.dt, edge_order=2)

    def fft_magnitude(self, series: np.ndarray) -> np.ndarray:
        self._spectrum_mode = True
        spectrum = np.fft.rfft(series)
        return np.abs(spectrum)

    def fft_phase(self, series: np.ndarray) -> np.ndarray:
        self._spectrum_mode = True
        spectrum = np.fft.rfft(series)
        return np.angle(spectrum)

    def band_pass(
        self,
        series: np.ndarray,
        flo: Optional[float] = None,
        fhi: Optional[float] = None,
        order: int = 4,
        zero_phase: bool = True,
    ) -> np.ndarray:
        """Butterworth filtering helper exposed as ``t()`` inside formulae.

        Parameters mirror the legacy Pascal behaviour:
        - ``flo`` / ``fhi`` may be given either as fractions (0-1) of Nyquist
          or as absolute frequencies in Hz.
        - ``order`` controls the Butterworth order (default 4).
        - ``zero_phase`` mimics the zero-phase toggle (1/0) by choosing
          between ``filtfilt`` and ``lfilter``.
        """

        if not _SCIPY_AVAILABLE:
            raise ValueError(
                "Butterworth filtering requires SciPy. Install scipy or adjust the formula."
            )

        if self.dt <= 0:
            raise ValueError("Cannot filter histories with zero or undefined time step")

        nyquist = 0.5 / self.dt
        if nyquist <= 0:
            raise ValueError("Invalid time step for filtering")

        def _normalise(freq: Optional[float]) -> Optional[float]:
            if freq is None:
                return None
            if freq <= 0:
                return 0.0
            if freq <= 1:
                # Treat as fraction of Nyquist (Pascal percentage style)
                return min(freq, 0.999)
            return min(freq / nyquist, 0.999)

        lo = _normalise(flo)
        hi = _normalise(fhi)

        if lo is None or lo <= 0:
            if hi is None or hi <= 0:
                raise ValueError("Provide cutoff frequency/frequencies for t()")
            btype = 'lowpass'
            wn = min(hi, 0.999)
        elif hi is None or hi <= 0:
            btype = 'highpass'
            wn = max(lo, 1e-6)
        else:
            if hi <= lo:
                raise ValueError("High-cut frequency must exceed low-cut frequency")
            btype = 'bandpass'
            wn = [max(lo, 1e-6), min(hi, 0.999)]

        b, a = butter(max(order, 1), wn, btype=btype)

        use_zero_phase = bool(zero_phase)

        if use_zero_phase:
            return filtfilt(b, a, series)
        return lfilter(b, a, series)

    def axis_for_result(self, values: np.ndarray) -> np.ndarray:
        if self._spectrum_mode:
            freq = np.fft.rfftfreq(len(self.time_axis), d=self.dt if self.dt else 1.0)
            self.axis_label = "frequency (Hz)"
            return freq.astype(np.float32)
        return self.time_axis.astype(np.float32)


def _ensure_1d(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array.astype(np.float32)
