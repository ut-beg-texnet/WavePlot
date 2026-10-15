"""IO layer against the cg11 dataset (3-D dumps, stopes). Skipped unless WAVEPLOT_TEST_DATA_B is set."""

from __future__ import annotations

import numpy as np
import pytest

from waveplot.io.binsource import read_snapshot
from waveplot.io.mapfile import read_map


def test_counts_and_sizes(sample_b_path):
    idx = read_map(sample_b_path)
    assert idx.nrecords == 274
    assert (len(idx.snaps), len(idx.dumps), len(idx.hists), len(idx.geoms)) == (211, 61, 1, 1)
    folder = sample_b_path.parent
    for ext in ("snp", "dmp", "hst", "geo"):
        assert idx.expected_sizes[ext] == (folder / f"cg11.{ext}").stat().st_size


def test_snapshots_match_headers(ds_b):
    for rec in ds_b.snapshots:
        a = read_snapshot(ds_b.snp, rec)
        assert a.max() == pytest.approx(rec.maxval, rel=1e-6)
        assert a.min() == pytest.approx(rec.minval, rel=1e-6)


def test_dump_planes(ds_b):
    planes = {s.slug: s for s in ds_b.dump_series.values()}
    assert set(planes) == {"ij|k=61", "kj|i=281"}
    ij, kj = planes["ij|k=61"], planes["kj|i=281"]
    assert ij.shape == (559, 299) and kj.shape == (119, 299)
    assert len(ij) + len(kj) == 61
    assert ij.dim3 and kj.dim3
    raw = ij.raw(len(ij) - 1)
    assert raw.shape == (559, 299, 9) and np.isfinite(raw).all()
    # stresses are much larger than velocities
    assert np.abs(raw[..., 3:]).max() > 1e3 * np.abs(raw[..., :3]).max()


def test_geometry_and_stope_links(ds_b):
    g = ds_b.geometry
    assert g.grid.as_tuple() == (1, 561, 1, 301, 1, 121)
    assert g.physical_size == (1400.0, 750.0, 300.0)
    assert g.boundary_stress["s11"] == pytest.approx(-21.86e6)
    (st,) = g.stopes
    assert st.stope_id == 2 and st.plane == "kj" and st.params["friction_deg"] == pytest.approx(30)
    linked = [s for s in ds_b.series.values() if s.stope_id == 2]
    assert len(linked) == 16 and all(s.is_stope for s in linked)


def test_history_sample_interval(ds_b):
    (h,) = ds_b.histories
    assert h.name == "MaxVel" and h.n == 301
    assert h.dt == pytest.approx(100 * 3.55099e-4, rel=1e-3)  # log prints the minimum timestep
