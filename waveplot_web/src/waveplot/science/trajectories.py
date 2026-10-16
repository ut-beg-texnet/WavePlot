"""Particle-motion trajectories from velocity component histories (``DI_ReadTraject``)."""

from __future__ import annotations

import numpy as np

from .compat import Mode
from .signal import integrate


def trajectory(vx: np.ndarray | None, vy: np.ndarray | None, vz: np.ndarray | None, dt: float, kind: str = "disp",
               mode: Mode = "standard", scale: float = 1.0) -> np.ndarray:
    """(n, 3) path of displacement (``disp``, integrated), velocity (``vel``) or acceleration (``acc``).

    v407 integrates with a rectangle rule (Σ v·dt) and differentiates as (v_prev − v)/dt, i.e. with
    the opposite sign; standard mode uses the trapezoid rule and np.gradient.
    """
    comps = [c for c in (vx, vy, vz) if c is not None]
    n = min(len(c) for c in comps) if comps else 0
    out = np.zeros((n, 3))
    for k, c in enumerate((vx, vy, vz)):
        if c is None:
            continue
        v = np.asarray(c, float)[:n] * scale
        if kind == "disp":
            out[:, k] = np.cumsum(v) * dt if mode == "v407" else integrate(v, dt)
        elif kind == "acc":
            if mode == "v407":
                prev = np.concatenate([[0.0], v[:-1]])
                out[:, k] = (prev - v) / dt
            else:
                out[:, k] = np.gradient(v, dt)
        else:
            out[:, k] = v
    return out
