"""Component rotations and combinations of multi-component histories."""

from __future__ import annotations

import math

import numpy as np

from .compat import Mode, arctan0


def rotate2(x: np.ndarray, y: np.ndarray, theta_deg: float) -> np.ndarray:
    """``C,θ``: x·cosθ + y·sinθ."""
    t = math.radians(theta_deg)
    return np.asarray(x, float) * math.cos(t) + np.asarray(y, float) * math.sin(t)


def hypot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(np.asarray(x, float), np.asarray(y, float))


def angle_deg(x: np.ndarray, y: np.ndarray, mode: Mode = "standard") -> np.ndarray:
    """``r``: direction of (x, y) in degrees, (−180, 180]."""
    if mode == "v407":
        return np.degrees(arctan0(np.asarray(y, float), np.asarray(x, float)))
    return np.degrees(np.arctan2(np.asarray(y, float), np.asarray(x, float)))


def vector_magnitude(*comps: np.ndarray) -> np.ndarray:
    return np.sqrt(sum(np.asarray(c, float) ** 2 for c in comps))


def source_angles(rec_ijk: tuple[int, int, int], src_ijk: tuple[float, float, float],
                  spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), mode: Mode = "standard"
                  ) -> tuple[float, float]:
    """(γ, θ): azimuth in the i–k plane and elevation towards j, from the source to the receiver.

    v407 uses grid-index differences (wrong when dx, dy, dz differ); standard uses metres.
    """
    di, dj, dk = (rec_ijk[0] - src_ijk[0], rec_ijk[1] - src_ijk[1], rec_ijk[2] - src_ijk[2])
    if mode == "standard":
        di, dj, dk = di * spacing[0], dj * spacing[1], dk * spacing[2]
        gam = math.atan2(dk, di)
        thet = math.atan2(dj, math.hypot(di, dk))
    else:
        gam = float(arctan0(dk, di))
        thet = float(arctan0(dj, math.sqrt(di * di + dk * dk)))
    return gam, thet


def rotate_to_source(x: np.ndarray, y: np.ndarray, z: np.ndarray, gam: float, thet: float, comp: int
                     ) -> np.ndarray:
    """``&R``: 0 radial (along the ray), 1 SV-like (in the vertical plane), 2 transverse (horizontal)."""
    cg, sg, ct, st = math.cos(gam), math.sin(gam), math.cos(thet), math.sin(thet)
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    if comp == 0:
        return x * cg * ct + y * st + z * sg * ct
    if comp == 1:
        return -x * cg * st + y * ct - z * sg * st
    return -x * sg + z * cg
