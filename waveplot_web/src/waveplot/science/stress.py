"""Stress-tensor quantities for dumps (tension positive, as in WAVE).

v407 (DATA_RSC.PAS:1154-1179): principal stresses of the in-plane tensor, excess shear stress
(ESS) on a plane rotated by θ with friction angle φ, and isochromatics (max in-plane shear).
The v407 ESS shear term has the opposite sign to a consistent tensor rotation, so shear and
normal stress refer to different planes; ``mode="standard"`` uses the consistent rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .compat import Mode, arctan0


@dataclass
class Principal2D:
    sigma_max: np.ndarray      # most tensile in-plane principal stress
    sigma_min: np.ndarray      # most compressive in-plane principal stress
    angle_min: np.ndarray      # direction of sigma_min, radians from ax1 towards ax2

    @property
    def angle_max(self) -> np.ndarray:
        return self.angle_min + np.pi / 2


def principal_2d(s11, s22, s12, mode: Mode = "standard") -> Principal2D:
    s11, s22, s12 = (np.asarray(a, dtype=float) for a in (s11, s22, s12))
    summ, diff = s11 + s22, s11 - s22
    r = np.sqrt(s12**2 + 0.25 * diff**2)
    # Direction of the most compressive axis in the data frame (ax1 → ax2). v407 computes
    # Ang = ½·Arctan0(2·s12, −Diff) and draws the bar at −Ang in screen coordinates, which is the
    # same physical direction; only Arctan0's (0, 0) edge case differs.
    if mode == "v407":
        ang = -0.5 * np.asarray(arctan0(2 * s12, -diff))
    else:
        ang = 0.5 * np.arctan2(-2 * s12, -diff)
    return Principal2D(0.5 * summ + r, 0.5 * summ - r, ang)


def max_shear_2d(s11, s22, s12) -> np.ndarray:
    """Isochromatics / max in-plane shear √(s12² + ¼(s11−s22)²)."""
    s11, s22, s12 = (np.asarray(a, dtype=float) for a in (s11, s22, s12))
    return np.sqrt(s12**2 + 0.25 * (s11 - s22) ** 2)


def rotated(s11, s22, s12, theta_deg: float, mode: Mode = "standard") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(σ11', σ22', τ12') in axes rotated by θ (v407 uses the opposite shear sign)."""
    t = math.radians(theta_deg)
    c2, s2, sc = math.cos(t) ** 2, math.sin(t) ** 2, math.sin(t) * math.cos(t)
    s11, s22, s12 = (np.asarray(a, dtype=float) for a in (s11, s22, s12))
    n11 = s11 * c2 + 2 * s12 * sc + s22 * s2
    n22 = s11 * s2 - 2 * s12 * sc + s22 * c2
    sign = 1.0 if mode == "v407" else -1.0
    t12 = sign * 0.5 * (s11 - s22) * math.sin(2 * t) + s12 * math.cos(2 * t)
    return n11, n22, t12


def ess(s11, s22, s12, theta_deg: float, friction_deg: float, cohesion: float = 0.0,
        mode: Mode = "standard") -> np.ndarray:
    """Excess shear stress on the plane with normal along the rotated ax2 axis:

    ESS = |τ'| − tanφ·|min(σn', 0)| − c   (compression reduces the available shear).
    """
    _n11, n22, t12 = rotated(s11, s22, s12, theta_deg, mode)
    sn = np.minimum(n22, 0.0)
    return np.abs(t12) - abs(math.tan(math.radians(friction_deg))) * np.abs(sn) - cohesion


def ess_optimal(s11, s22, s12, friction_deg: float, cohesion: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Maximum ESS over all plane orientations (exact, over the Mohr circle) and the angle (degrees,
    from ax1) of the corresponding plane normal.

    Planes map to points σn = σm + R·c, |τ| = R·√(1−c²), c ∈ [−1, 1]. Where σn ≥ 0 the ESS is |τ|;
    where σn < 0 it is |τ| + μ·σn, maximised at c* = μ/√(1+μ²) with value μ·σm + R·√(1+μ²).
    """
    s11, s22, s12 = (np.asarray(a, dtype=float) for a in (s11, s22, s12))
    mu = abs(math.tan(math.radians(friction_deg)))
    sm = 0.5 * (s11 + s22)
    r = np.sqrt(s12**2 + 0.25 * (s11 - s22) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        c0 = np.where(r > 0, -sm / r, np.where(sm >= 0, -np.inf, np.inf))   # σn ≥ 0 for c ≥ c0
    cstar = mu / math.sqrt(1 + mu * mu)
    # region σn ≥ 0 (c ≥ c0): max |τ| at c = clip(0, c0, 1)
    ca = np.clip(np.maximum(c0, 0.0), -1.0, 1.0)
    a_val = np.where(c0 <= 1.0, r * np.sqrt(np.maximum(1 - ca**2, 0.0)), -np.inf)
    # region σn < 0 (c < c0): unconstrained optimum if c* < c0, otherwise the boundary (same as region A)
    b_val = np.where(cstar < c0, mu * sm + r * math.sqrt(1 + mu * mu), -np.inf)
    best = np.maximum(a_val, b_val)
    c_opt = np.where(b_val >= a_val, cstar, ca)
    p = principal_2d(s11, s22, s12)
    normal = np.degrees(p.angle_max + 0.5 * np.arccos(np.clip(c_opt, -1, 1))) % 180
    return best - cohesion, normal


@dataclass
class Principal3D:
    values: np.ndarray        # (..., 3) descending: σ_max (most tensile), σ_int, σ_min (most compressive)
    vectors: np.ndarray       # (..., 3, 3) columns = directions (i, j, k components)


def principal_3d(s11, s22, s33, s12, s23, s13) -> Principal3D:
    s11, s22, s33, s12, s23, s13 = (np.asarray(a, dtype=float) for a in (s11, s22, s33, s12, s23, s13))
    T = np.empty((*s11.shape, 3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = s11, s22, s33
    T[..., 0, 1] = T[..., 1, 0] = s12
    T[..., 1, 2] = T[..., 2, 1] = s23
    T[..., 0, 2] = T[..., 2, 0] = s13
    w, v = np.linalg.eigh(T)
    return Principal3D(w[..., ::-1], v[..., ::-1])


def mean_stress(s11, s22, s33=None) -> np.ndarray:
    if s33 is None:
        return 0.5 * (np.asarray(s11, float) + np.asarray(s22, float))
    return (np.asarray(s11, float) + np.asarray(s22, float) + np.asarray(s33, float)) / 3


def von_mises(s11, s22, s33, s12, s23, s13) -> np.ndarray:
    s11, s22, s33, s12, s23, s13 = (np.asarray(a, dtype=float) for a in (s11, s22, s33, s12, s23, s13))
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12**2 + s23**2 + s13**2))


@dataclass
class MohrCircles:
    circles: list[tuple[float, float]]         # (centre, radius)
    points: list[tuple[float, float, str]]     # (σn, τ, label)
    principal: list[float]


def mohr(s11: float, s22: float, s12: float, s33: float | None = None, s23: float = 0.0, s13: float = 0.0
         ) -> MohrCircles:
    if s33 is None:
        c, r = 0.5 * (s11 + s22), math.sqrt(s12**2 + 0.25 * (s11 - s22) ** 2)
        return MohrCircles([(c, r)], [(s11, s12, "σ11, τ12"), (s22, -s12, "σ22, −τ12")], [c + r, c - r])
    p = principal_3d(s11, s22, s33, s12, s23, s13).values
    s1, s2, s3 = (float(x) for x in p)
    circles = [((s1 + s3) / 2, (s1 - s3) / 2), ((s1 + s2) / 2, (s1 - s2) / 2), ((s2 + s3) / 2, (s2 - s3) / 2)]
    return MohrCircles(circles, [(s11, s12, "σ11, τ12"), (s22, -s12, "σ22, −τ12")], [s1, s2, s3])


def failure_line(friction_deg: float, cohesion: float = 0.0, sigma_range: tuple[float, float] = (-1, 0)
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Coulomb envelope |τ| = c − tanφ·σn for compressive σn (tension positive)."""
    sn = np.linspace(min(sigma_range[0], 0.0), 0.0, 50)
    return sn, cohesion - math.tan(math.radians(friction_deg)) * sn
