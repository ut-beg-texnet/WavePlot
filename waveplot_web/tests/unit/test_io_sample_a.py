"""IO layer against the in-repo sample dataset (Waveplot_exe/mt_slip5)."""

from __future__ import annotations

import numpy as np
import pytest

from waveplot.io.binsource import read_history, read_snapshot, snapshot_marker_ok
from waveplot.io.mapfile import read_map


def test_record_counts(sample_a_path):
    idx = read_map(sample_a_path)
    assert idx.nrecords == 237
    assert (len(idx.snaps), len(idx.hists), len(idx.geoms), len(idx.dumps), len(idx.cracks)) == (160, 76, 1, 0, 0)
    assert idx.issues == []


def test_offsets_match_file_sizes(sample_a_path):
    idx = read_map(sample_a_path)
    folder = sample_a_path.parent
    assert idx.expected_sizes["snp"] == (folder / "mt_slip5.snp").stat().st_size
    assert idx.expected_sizes["hst"] == (folder / "mt_slip5.hst").stat().st_size
    assert idx.expected_sizes["geo"] == (folder / "mt_slip5.geo").stat().st_size


def test_every_snapshot_matches_header_extremes(ds_a):
    """Strongest check of offsets/slicing: recomputed max/min equal the .map header values."""
    for rec in ds_a.snapshots:
        a = read_snapshot(ds_a.snp, rec)
        assert a.shape == (rec.xqty, rec.yqty)
        assert a.max() == pytest.approx(rec.maxval, rel=1e-6)
        assert a.min() == pytest.approx(rec.minval, rel=1e-6)
        assert snapshot_marker_ok(ds_a.snp, rec)


def test_every_history_matches_header_extremes(ds_a):
    for rec in ds_a.histories:
        a = read_history(ds_a.hst, rec)
        assert len(a) == rec.n == 4001
        assert a.max() == pytest.approx(rec.maxval, rel=1e-6)
        assert a.min() == pytest.approx(rec.minval, rel=1e-6)


def test_snapshot_layout(ds_a):
    names = {(s.name, s.axes, s.fixed_index) for s in ds_a.snapshots}
    assert names == {("V-abs", "ik", 50), ("S11", "ik", 50), ("S22", "ij", 250), ("S33", "kj", 250)}
    v = ds_a.snapshot_record(1)
    assert v.name == "V-abs" and (v.xqty, v.yqty) == (509, 509)
    times = sorted({round(s.time, 4) for s in ds_a.snapshots})
    assert len(times) == 40 and times[0] == pytest.approx(0.18, abs=1e-3) and times[-1] == pytest.approx(7.2001)


def test_series_grouping(ds_a):
    assert set(ds_a.series) == {"V-abs|ik|j=50", "S11|ik|j=50", "S22|ij|k=250", "S33|kj|i=250"}
    for s in ds_a.series.values():
        assert len(s) == 40
        assert np.all(np.diff(s.times) > 0)


def test_source_location_orientation(ds_a):
    """The source at grid (250, 50, 250) must appear near row i=250, column k=250 of the ik plane."""
    s = ds_a.series["S11|ik|j=50"]
    a1, a2 = s.coords()
    energy = s.frame(0).astype(np.float64) ** 2
    ci = (energy.sum(axis=1) * a1.index).sum() / energy.sum()
    ck = (energy.sum(axis=0) * a2.index).sum() / energy.sum()
    assert abs(ci - 250) <= 3
    assert abs(ck - 250) <= 3
    # the S22 i-j plane (k=250) carries the source at j=50
    s22 = ds_a.series["S22|ij|k=250"]
    _b1, b2 = s22.coords()
    e2 = s22.frame(0).astype(np.float64) ** 2
    cj = (e2.sum(axis=0) * b2.index).sum() / e2.sum()
    assert abs(cj - 50) <= 5


def test_histories_time_base(ds_a):
    h = ds_a.history_record(2)
    assert h.name == "S11" and h.position == (100, 85, 100)
    assert h.tstart == 0.0 and h.tend == pytest.approx(7.2001, rel=1e-5)
    assert h.dt == pytest.approx(7.2001 / 4000, rel=1e-5)
    g = ds_a.history_record(1)
    assert g.name == "MaxVel" and g.is_global


def test_receivers_match_script(ds_a):
    sc = ds_a.script
    expected = {h.position for h in sc.histories if h.position}
    got = {r.position for r in ds_a.receivers}
    assert got == expected and len(got) == 25
    assert all(set(r.names) == {"S11", "S22", "S33"} for r in ds_a.receivers)


def test_geometry(ds_a):
    g = ds_a.geometry
    assert g is not None and g.model3d
    assert g.grid.as_tuple() == (1, 511, 1, 101, 1, 511)
    assert g.physical_size == (10200.0, 2000.0, 10200.0)
    assert g.spacing == {"i": 20.0, "j": 20.0, "k": 20.0}
    sc_props = {p.material: (p.vp, p.vs) for p in ds_a.script.props}
    for p in g.props:
        vp, vs = sc_props[p.number]
        assert p.vp == pytest.approx(vp, rel=1e-5) and p.vs == pytest.approx(vs, rel=1e-5)
    # material boxes match the .m 'mat' lines; zones lo..hi span [(lo-1)d, hi*d]
    assert [m.number for m in g.materials] == [1, 2, 3, 3, 4, 4]
    m2 = g.materials[1]
    assert m2.extent.as_tuple() == (10, 500, 80, 99, 10, 500)
    assert m2.bounds_m("i", 20.0) == (180.0, 10000.0)
    # sources: reverse order restored, components = WAVE variable numbers
    assert [s.component for s in g.sources] == ["S11", "S22", "S33", "S12", "S31", "S23"]
    assert g.sources[0].params["period"] == pytest.approx(0.1)
    assert g.source_centroid() == (250, 50, 250)


def test_run_log_and_info(ds_a):
    assert ds_a.runlog.timestep == pytest.approx(0.0018)
    assert ds_a.runlog.final_cycle == (4000, pytest.approx(7.20014))
    info = ds_a.info()
    assert info["grid_cells"] == [510, 100, 510]
    assert info["counts"]["receivers"] == 25


def test_xarray_views(ds_a):
    da = ds_a.snapshot(2)
    assert da.dims == ("x", "z") and da.attrs["units"] == "Pa"
    assert float(da.x[0]) == pytest.approx(10.0)       # i=1 cell centre at 0.5*dx
    h = ds_a.history(3)
    assert h.dims == ("time",) and h.attrs["position_ijk"] == [100, 85, 100]


def test_cell_series_matches_frames(ds_a):
    s = ds_a.series["S22|ij|k=250"]
    cs = s.cell_series(200, 50, frames=[0, 5, 10])
    for n, f in enumerate([0, 5, 10]):
        assert cs[n] == s.frame(f)[200, 50]


def test_envelope(ds_a):
    s = ds_a.series["S33|kj|i=250"]
    env = s.envelope("max", frames=[0, 1, 2])
    stack = np.stack([s.frame(f) for f in (0, 1, 2)])
    np.testing.assert_array_equal(env, stack.max(axis=0))
    env_abs = s.envelope("absmax", frames=[0, 1, 2])
    np.testing.assert_array_equal(np.abs(env_abs), np.abs(stack).max(axis=0))
