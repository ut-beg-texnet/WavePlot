from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from waveplot.science.colormaps import CMAPS, banded_colorscale, get_colors
from waveplot.science.compat import arctan0, pas_pow, pas_round, round_places_up
from waveplot.science.decimate import coords_for, dump_glyph_indices, fit_grid
from waveplot.science.levels import compute_levels


def pascal_calc_levels(maxval, minval, q, fill, absval=False, locked=False):
    """Loop transliteration of CalcLevels (non-log), DSPSNAP.PAS:62-108."""
    import math

    lev_array = (0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.79, 0.86, 0.91, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 1)
    if not locked:
        rat = (q - 0.5) / q
        maxval = maxval * rat if maxval >= 0 else maxval / rat
        minval = minval * rat if minval <= 0 else minval / rat
    if absval:
        maxval, minval = max(abs(maxval), abs(minval)), max(0, minval)
    if maxval == minval:
        maxval = minval + 1
    level = [0.0] * (q + 1)
    lev0 = 1
    for lev in range(1, q + 1):
        x = (lev - 1) / (q - 1)
        if fill == "lin":
            level[lev] = x * (maxval - minval) + minval
        elif fill == "quad":
            level[lev] = x * x * (maxval - minval) + minval
        elif fill == "fixed":
            level[lev] = lev_array[math.trunc(x * 15)] * maxval
        elif fill == "dquad":
            level[lev] = (x * x) ** 2 * (maxval - minval) + minval
        elif fill == "cos":
            level[lev] = 0.5 * ((1 - math.cos(math.pi * x)) ** 2) ** 2 * (maxval - minval) + minval
        if lev > 1 and abs(level[lev]) < abs(level[lev0]):
            lev0 = lev
    level[lev0] = 0
    return level[1:]


@pytest.mark.parametrize("fill", ["lin", "quad", "dquad", "cos", "fixed"])
@pytest.mark.parametrize("rng", [(-0.656, 1.69), (2.0, 5.0), (-3.0, -1.0), (-1.0, 1.0)])
@pytest.mark.parametrize("absval", [False, True])
def test_v407_levels_match_transliteration(fill, rng, absval):
    lo, hi = rng
    got = compute_levels(lo, hi, 11, fill, abs_values=absval, mode="v407")
    np.testing.assert_allclose(got.values, pascal_calc_levels(hi, lo, 11, fill, absval), rtol=1e-12, atol=1e-15)
    assert got.values[got.zero_index] == 0
    assert "level_shrink" in got.provenance.quirks


def test_v407_log_levels_symmetric_and_zero():
    lev = compute_levels(-0.656, 1.69, 11, "log", mode="v407")
    v = lev.values
    assert v[lev.zero_index] == 0
    assert v[-1] == pytest.approx(1.69 * 10.5 / 11)
    # geometric progression away from zero on the positive side
    pos = v[lev.zero_index + 1:]
    np.testing.assert_allclose(pos[1:] / pos[:-1], lev.ratio, rtol=1e-9)


def test_band_toward_zero():
    lev = compute_levels(-1, 1, 5, "lin", mode="v407", locked=True)   # [-1, -0.5, 0, 0.5, 1]
    np.testing.assert_allclose(lev.values, [-1, -0.5, 0, 0.5, 1])
    z = np.array([-0.9, -0.5, -0.2, 0.0, 0.2, 0.7, 1.5, -2.0])
    np.testing.assert_array_equal(lev.band_index(z), [1, 1, 2, 2, 2, 3, 4, 0])


def test_standard_levels():
    lev = compute_levels(-2, 6, 9, "lin")
    np.testing.assert_allclose(lev.values, np.linspace(-2, 6, 9))
    cos = compute_levels(0, 1, 5, "cos").values
    assert cos[0] == 0 and cos[-1] == pytest.approx(1) and np.all(np.diff(cos) > 0)
    sl = compute_levels(-1e-3, 2.0, 11, "log").values
    assert sl[0] == pytest.approx(-1e-3) and sl[-1] == pytest.approx(2.0) and np.all(np.diff(sl) > 0)


def test_colormaps_and_banded_scale():
    for name in CMAPS:
        cols = get_colors(name, 16)
        assert all(c.startswith("#") and len(c) == 7 for c in cols)
    stops = banded_colorscale(np.array([0.0, 1.0, 2.0]), ["#000000", "#ff0000", "#00ff00"], 0.0, 3.0)
    assert stops[0] == [0.0, "#000000"] and stops[-1][0] == 1.0
    assert all(stops[k][0] <= stops[k + 1][0] for k in range(len(stops) - 1))


def test_pascal_primitives():
    assert pas_round(2.5) == 3 and pas_round(-2.5) == -3 and pas_round(0.49) == 0
    assert arctan0(0.0, 0.0) == pytest.approx(-np.pi / 2)
    assert arctan0(1.0, 0.0) == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(pas_pow(np.array([-8.0, 8.0]), 3.0), [-512, 512])
    np.testing.assert_allclose(pas_pow(np.array([-8.0]), 1 / 3), [2.0])      # v407 drops the sign
    np.testing.assert_allclose(pas_pow(np.array([-4.0]), 0.5), [2.0])
    assert round_places_up(1.234, 2) == pytest.approx(1.24)
    assert round_places_up(0.0123, 2) == pytest.approx(0.013)


def test_fit_grid_methods():
    a = np.arange(36, dtype=np.float32).reshape(6, 6)
    m = fit_grid(a, (3, 3), "mean")
    np.testing.assert_allclose(m[0, 0], a[:2, :2].mean())
    p = fit_grid(np.array([[1, -5, 2, 0]], np.float32).T, (2, 1), "peak")
    assert p.ravel().tolist() == [-5, 2]
    n = fit_grid(a, (2, 2), "nearest")
    assert n.shape == (2, 2)
    v = fit_grid(a, (3, 6), "v407")
    np.testing.assert_allclose(v[:, 0], [0, (6 + 12 + 18) / 3, (24 + 30) / 2])   # trailing blocks, first exact
    assert fit_grid(a, (10, 10)) is a
    assert len(coords_for(np.arange(6.0), 3)) == 3


def test_dump_glyph_indices():
    assert list(dump_glyph_indices(10, 5, "v407")) == [0, 2, 4, 6, 8]   # 1-based 1,3,5,7,9
    assert list(dump_glyph_indices(10, 5)) == [1, 3, 5, 7, 9]
    assert len(dump_glyph_indices(4, 10)) == 4


def test_lazy_datatree(sample_a_path, ds_a):
    tree = xr.open_datatree(sample_a_path, engine="wave")
    v = tree["snapshots/S33_kj_i250"]["S33"]
    assert v.shape == (40, 511, 101)
    np.testing.assert_array_equal(v.isel(time=4).values, ds_a.series["S33|kj|i=250"].frame(4))
    np.testing.assert_array_equal(v.isel(time=[1, 2], z=slice(0, 3)).values[1],
                                  ds_a.series["S33|kj|i=250"].frame(2)[:3, :])   # dims (time, z=k, y=j)
    assert tree["histories"]["value"].shape == (76, 4001)


def test_netcdf_roundtrip(tmp_path, ds_a):
    from waveplot.export.netcdf import to_netcdf

    p = to_netcdf(ds_a, tmp_path / "a.nc", include=["histories", "geometry"])
    t = xr.open_datatree(p, engine="h5netcdf")
    np.testing.assert_array_equal(t["histories"]["value"].values[1], ds_a.history_data(2))
    np.testing.assert_allclose(t["geometry"]["vp"].values[0], 5000.0, rtol=1e-5)
