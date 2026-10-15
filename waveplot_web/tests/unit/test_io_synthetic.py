"""IO layer against synthetic, spec-conformant files (formats without real samples)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fixtures.synth import SynthDataset
from waveplot import open_dataset
from waveplot.io.names import split_func, stagger_shift, var_info


def test_2d_dump_time_axis_snapshot_and_func_code(tmp_path):
    sd = SynthDataset()
    snap = np.arange(12, dtype=np.float32).reshape(3, 4)
    sd.snapshot(snap, var=3, time=0.5, axes="it", extent=(2, 4, 5, 5, 1, 1), dx=10.0, dy=0.001, t1=0.1,
                func=1)
    dump = np.random.default_rng(1).normal(size=(4, 3, 5)).astype(np.float32)
    sd.dump(dump, 1.25, extent=(1, 4, 1, 3, 1, 1), iqty=4, jqty=3, kqty=1, write_kqty=False)
    sd.history(np.linspace(0, 1, 11), var=1, t1=0.0, t2=1.0, pos=(3, 4, 1))
    path = sd.write(tmp_path)
    ds = open_dataset(path)
    assert ds.issues == []
    s = ds.snapshots[0]
    assert s.func == 1 and s.name == "S11 MX" and s.is_spacetime
    np.testing.assert_array_equal(ds.series[next(iter(ds.series))].frame(0), snap)
    a1, a2 = ds.series[next(iter(ds.series))].coords()
    assert list(a1.index) == [2, 3, 4] and a2.axis == "t"
    assert a2.coord[0] == pytest.approx(0.1) and a2.coord[-1] == pytest.approx(0.5)
    d = ds.dumps[0]
    assert not d.dim3 and d.kqty == 1 and d.axes == "ij"
    ser = next(iter(ds.dump_series.values()))
    np.testing.assert_array_equal(ser.raw(0), dump)
    ip = ser.in_plane(0)
    np.testing.assert_array_equal(ip["s12"], dump[..., 4])


def test_kj_dump_plane_detection(tmp_path):
    sd = SynthDataset()
    data = np.zeros((5, 3, 9), np.float32)  # X=k (5), Y=j (3)
    data[..., 8] = 7.0                       # s13
    data[..., 7] = 3.0                       # s23
    sd.dump(data, 0.1, extent=(9, 9, 1, 3, 1, 5), iqty=1, jqty=3, kqty=5)
    ds = open_dataset(sd.write(tmp_path))
    d = ds.dumps[0]
    assert d.dim3 and d.axes == "kj" and d.fixed_axis == "i" and d.fixed_index == 9
    ip = next(iter(ds.dump_series.values())).in_plane(0)
    assert ip["s12"].mean() == 3.0  # k-j plane uses s23 as in-plane shear


def test_upper_case_extensions_and_truncated_map(tmp_path):
    sd = SynthDataset()
    sd.snapshot(np.ones((2, 2)), var=4, time=1.0)
    path = sd.write(tmp_path, stem="RUN", upper=True)
    with open(path, "ab") as f:
        f.write(b"\x00" * 50)
    ds = open_dataset(tmp_path / "run")       # base name, different case
    assert len(ds.snapshots) == 1
    assert any("trailing bytes" in i.message for i in ds.issues)
    assert ds.series[next(iter(ds.series))].frame(0).tolist() == [[1, 1], [1, 1]]


def test_missing_data_is_reported(tmp_path):
    sd = SynthDataset()
    sd.snapshot(np.ones((2, 2)), var=4, time=1.0)
    path = sd.write(tmp_path)
    (tmp_path / "synth.snp").write_bytes(b"\x00" * 8)
    ds = open_dataset(path)
    assert any(i.severity == "error" for i in ds.issues)
    from waveplot.io.binsource import DataUnavailable

    with pytest.raises(DataUnavailable):
        ds.series[next(iter(ds.series))].frame(0)


def test_geometry_with_stope_layout_and_crack_dump(tmp_path):
    sd = SynthDataset()
    layout = np.zeros((4, 1, 6), np.uint8)    # (nk, nj, ni) for an i-k stope
    layout[1:3, 0, 2:5] = 1
    sd.geometry(props=[(3.82e10, 1.8225e10, 2500.0)], mats=[(1, 1, 10, 1, 10, 1, 10)],
                sources=[(3, 5, 5, 5, 5, 5, 5), (4, 6, 6, 5, 5, 5, 5)],
                stopes=[(1, 2, (3, 8, 4, 4, 2, 5), layout)], extent=(1, 10, 1, 10, 1, 10))
    vals = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    vals[1, 13] = 1.0   # Taslp > 0 -> slipping
    vals[2, 8] = 0.0    # sNs1 == 0 -> open
    sd.crack(0.5, [(1, (3, 8, 4, 4, 2, 2), vals)])
    ds = open_dataset(sd.write(tmp_path))
    g = ds.geometry
    assert g.issues == []
    assert g.props[0].vp == pytest.approx(5000.0, rel=1e-5)
    assert [s.component for s in g.sources] == ["S11", "S22"]
    assert g.source_centroid() == (6, 5, 5)   # v407 formula: (1+Σi1+1+Σi2) DIV 2N
    st_ = g.stopes[0]
    assert st_.plane == "ik" and st_.layout is not None
    np.testing.assert_array_equal(st_.layout, layout)
    elems = ds.crack_dump(1)
    assert len(elems) == 1 and elems[0].values.shape == (3, 16)
    np.testing.assert_array_equal(elems[0].values, vals)
    assert list(elems[0].state()) == [2, 1, 2] or list(elems[0].state())[1] == 1


def test_var_table():
    assert var_info(3).name == "S11" and var_info(3).units == "Pa"
    assert var_info(22).name == "V-abs" and var_info(22).units == "m/s"
    assert var_info(151).name == "MaxVel"
    assert var_info(60).name == "T1slip" and var_info(60).is_stope
    unk = var_info(15)
    assert not unk.known and unk.name == "VAR#15"
    assert split_func(3 + 2 * 65535) == (3, 2)
    assert stagger_shift(1, "i", "j") == (1, 0)       # XVEL shifted along i
    assert stagger_shift(3, "i", "k") == (1, 0)       # S12 on i-k plane: shift in i only
    assert stagger_shift(6, "k", "j") == (1, 1)       # S23 on k-j plane


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(nx=st.integers(1, 6), ny=st.integers(1, 6), nsnap=st.integers(1, 4), nh=st.integers(0, 3),
       seed=st.integers(0, 1000))
def test_round_trip(tmp_path_factory, nx, ny, nsnap, nh, seed):
    rng = np.random.default_rng(seed)
    sd = SynthDataset()
    snaps = [rng.normal(size=(nx, ny)).astype(np.float32) for _ in range(nsnap)]
    hists = [rng.normal(size=rng.integers(2, 20)).astype(np.float32) for _ in range(nh)]
    for n, a in enumerate(snaps):
        sd.snapshot(a, var=4, time=0.1 * (n + 1), extent=(1, nx, 1, ny, 3, 3))
    for h in hists:
        sd.history(h, var=1, t1=0.0, t2=1.0)
    ds = open_dataset(sd.write(tmp_path_factory.mktemp("rt")))
    assert ds.issues == []
    s = ds.series["S22|ij|k=3"]
    for n, a in enumerate(snaps):
        np.testing.assert_array_equal(s.frame(n), a)
    for n, h in enumerate(hists, start=1):
        np.testing.assert_array_equal(ds.history_data(n), h)
