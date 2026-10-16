# WavePlot Web

Interactive, browser-based visualisation of **WAVE** simulation outputs — a Python port and
modernisation of the Turbo Pascal **WavePlot v407** (`../Turbo pascal version/`).

* **Snapshots** — animated 2-D field planes: colour maps, contours, multi-line (wiggle) views,
  geometry and receiver overlays, multi-panel grids, a probe that plots any cell's time series,
  PNG/PDF/CSV/MP4/GIF exports.
* **Dumps** — principal-stress crosses, velocity arrows, ESS / max-shear / principal / von Mises
  maps, difference maps, Mohr circles.
* **Histories** — receiver time series with processing pipelines (modern expressions *or* the
  original v407 formula language), spectra, spectrograms, response spectra, record sections,
  metrics, legacy ASC/TBL exports.
* **Geometry** — 3-D model (materials, sources, faults, receivers, snapshot planes),
  cross-sections, particle motion, crack dumps.
* **Compare** — two runs side by side or as differences.
* **Standard / v407 numerics** — corrected maths by default; a switch reproduces the original
  program's calculations (and its quirks) exactly, with a badge listing what differs.

## Installation

Python 3.12+ (tested with 3.13). From the repository root:

```powershell
py -3.13 -m venv waveplot_python_venv
waveplot_python_venv\Scripts\python -m pip install --only-binary=:all: -e "waveplot_web[dev,server]" -c waveplot_web\constraints-py313.txt
```

(macOS / Linux: `python3 -m venv waveplot_python_venv` and `waveplot_python_venv/bin/python …`.)
All dependencies, including an ffmpeg binary for MP4 export (`imageio-ffmpeg`), are installed into
the virtual environment; nothing is installed globally.

## Running

```powershell
waveplot_python_venv\Scripts\waveplot serve Waveplot_exe            # a folder (searched for *.map)
waveplot_python_venv\Scripts\waveplot serve D:\runs\cg11\cg11.map    # a single run
waveplot_python_venv\Scripts\waveplot serve --data-root runs=D:\runs --port 8050 --prod
```

The browser opens at <http://localhost:8050/>. `--prod` serves with waitress (multi-threaded,
suitable for a lab server); without it Dash's development server is used. Datasets are only read
from the configured data roots and are addressed by `root/relative/path` ids.

Other commands:

```
waveplot info run.map [--json]                     summary of a dataset
waveplot convert run.map --to netcdf [-o run.nc]   self-describing netCDF4/HDF5 copy
waveplot formula run.map "[2]I&f,1,10" --mode v407 --explain [--out trace.csv]
waveplot export run.map --series "S11|ik|j=50" -o s11.mp4 [--frames 0:20] [--cmap balance]
waveplot settings import-legacy WVPL407.SET        import v407 preferences
```

## Python / Jupyter

```python
import waveplot, xarray as xr

ds = waveplot.open_dataset("Waveplot_exe/mt_slip5.map")
ds.summary()["series"]                                  # pandas tables
s11 = ds.series["S11|ik|j=50"].to_xarray()              # (time, x, z), coordinates in metres
h = ds.history(2)                                        # DataArray with a time axis
tree = xr.open_datatree("Waveplot_exe/mt_slip5.map", engine="wave")   # lazy access to everything

from waveplot.formula.evaluate import evaluate_text
r = evaluate_text(ds, "h2 | integrate | bandpass(1, 10)")
r.trace.x, r.trace.y
```

## Tests

```powershell
cd waveplot_web
..\waveplot_python_venv\Scripts\python -m pytest            # unit, golden (v407), web, CLI, smoke
$env:WAVEPLOT_TEST_DATA_B = "C:\path\to\cg11_inputs"         # optional second dataset (3-D dumps, faults)
..\waveplot_python_venv\Scripts\ruff check src tests
```

The golden tests compare the vectorised v407 implementations with loop transliterations of the
Pascal sources (`tests/golden/pascal_ref.py`).

## Layout

```
src/waveplot/
  io/        .map/.snp/.hst/.dmp/.geo/.crk readers, text side files, xarray backend, legacy settings
  model/     WaveDataset, snapshot series, receivers, dump planes, coordinates, dataset registry
  science/   signal processing, spectra, filters, stress, colour levels, colormaps, contours, v407 compat
  formula/   operation registry, legacy v407 parser, modern expressions, evaluator
  render/    Plotly figure builders (no Dash), matplotlib-free
  export/    CSV/ASC/TBL, netCDF, matplotlib figures and animations
  web/       Dash app: shell, pages, components, assets
docs/dev/    formats.md (binary formats), architecture.md, compat.md (v407 findings)
```
