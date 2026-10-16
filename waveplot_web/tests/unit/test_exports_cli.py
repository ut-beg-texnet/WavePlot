from __future__ import annotations

import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.conftest import REPO
from waveplot.cli import app as cli_app
from waveplot.export.tables import frame_csv, legacy_asc, legacy_tbl, traces_csv
from waveplot.formula.trace import Trace
from waveplot.io.legacy_settings import read_legacy_settings


def test_legacy_tables():
    a = Trace(np.array([1.0, -2.0, 3.0]), 0.5, 0.1, label="a")
    b = Trace(np.array([4.0, 5.0]), 0.5, 0.1, label="b")
    asc = legacy_asc([a, b], with_time=True).splitlines()
    assert len(asc) == 3 and len(asc[0]) == 3 * 14
    assert float(asc[1][:14]) == pytest.approx(0.6) and float(asc[1][14:28]) == pytest.approx(-2.0)
    assert asc[2][28:].strip() == ""
    assert legacy_asc([a, b], with_time=False, mode="v407").count("\n") == 3
    tbl = legacy_tbl(a).splitlines()
    assert tbl[0] == "3" and float(tbl[2]) == pytest.approx(-2.0)
    csv = traces_csv([a, b]).splitlines()
    assert csv[0].startswith("a s,a") and len(csv) == 4


def test_frame_csv(ds_a):
    s = ds_a.series["S22|ij|k=250"]
    text = frame_csv(s, 0)
    data = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=2)
    assert data.shape == (511 * 101, 5)
    assert data[:, 4].max() == pytest.approx(s.first.maxval, rel=1e-6)


def test_images_and_animation(tmp_path, ds_a):
    from waveplot.export.images import export_series, render_panel_bytes
    from waveplot.render.panels import snapshot_panel
    from waveplot.render.snapshot_fig import SnapshotStyle

    s = ds_a.series["S33|kj|i=250"]
    png = render_panel_bytes(snapshot_panel(ds_a, s, 3), SnapshotStyle(), ds_a, "png", dpi=80)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pdf = render_panel_bytes(snapshot_panel(ds_a, s, 3), SnapshotStyle(representation="contour"), ds_a, "pdf")
    assert pdf[:4] == b"%PDF"
    out = export_series(ds_a, s, tmp_path / "a.mp4", frames=[0, 1, 2, 3], fps=4, dpi=50)
    import imageio_ffmpeg

    nframes, _secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert nframes == 4
    gif = export_series(ds_a, s, tmp_path / "a.gif", frames=[0, 1], fps=2, dpi=40)
    assert gif.read_bytes()[:3] == b"GIF"


@pytest.mark.parametrize("path", ["Turbo pascal version/WVPL407.SET", "Waveplot_exe/WVPL407.SET"])
def test_legacy_settings(path):
    p = REPO / path
    if not p.is_file():
        pytest.skip("settings file not in checkout")
    s = read_legacy_settings(p)
    assert s.qty_levels > 0 and s.snap_fill in ("Log", "Quad", "Lin", "Fixed", "Cosn", "DQuad", "SqLog")
    if path.startswith("Waveplot_exe"):
        assert (s.drive, s.directory, s.fname) == ("C", ".", "WAVE")
        assert s.qty_levels == 11 and s.leg_max[0] == pytest.approx(1e6)
    else:
        assert s.fname == "l3d_5a3" and s.theta == pytest.approx(45) and s.friction == pytest.approx(30)


def test_cli(sample_a_path, tmp_path):
    runner = CliRunner()
    r = runner.invoke(cli_app, ["info", str(sample_a_path)])
    assert r.exit_code == 0, r.output
    assert "snapshots 160 in 4 series" in r.output
    r = runner.invoke(cli_app, ["formula", str(sample_a_path), "[2]I&f,1,10", "--mode", "v407", "--explain"])
    assert r.exit_code == 0, r.output
    assert "h2 | integrate | bandpass" in r.output
    r = runner.invoke(cli_app, ["export", str(sample_a_path), "--series", "S22|ij|k=250", "-o",
                                str(tmp_path / "f.png"), "--frames", "2:3"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "f.png").stat().st_size > 1000
    r = runner.invoke(cli_app, ["--version"])
    assert "waveplot" in r.output


def test_server_smoke(sample_a_path):
    """Start ``waveplot serve --prod`` in a subprocess and request a page and the API."""
    exe = Path(sys.executable).with_name("waveplot.exe" if sys.platform == "win32" else "waveplot")
    cmd = [str(exe), "serve", str(sample_a_path.parent), "--no-open", "--port", "0", "--prod"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = None
        t0 = time.time()
        while time.time() - t0 < 60:
            line = proc.stdout.readline()
            if "open http" in line:
                url = line.split("open", 1)[1].strip()
                break
        assert url, "server did not report its URL"
        for _ in range(50):
            try:
                with urllib.request.urlopen(url + "api/v1/health", timeout=2) as resp:
                    assert resp.status == 200
                    break
            except OSError:
                time.sleep(0.2)
        with urllib.request.urlopen(url, timeout=10) as resp:
            assert resp.status == 200 and b"WavePlot" in resp.read()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
