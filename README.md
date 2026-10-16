# WavePlot

Visualisation and analysis tool for **WAVE** simulation outputs (`.map` + `.snp/.hst/.dmp/.geo`),
originally written in Turbo Pascal (WavePlot v407, `Turbo pascal version/`).

## WavePlot Web (`waveplot_web/`)

A browser-based port and modernisation of v407: snapshots, dumps (stress tensors), histories with
processing pipelines and the original formula language, 3-D geometry, run comparison, exports, and a
Standard / v407 numerics switch that reproduces the original calculations exactly.

```powershell
py -3.13 -m venv waveplot_python_venv
waveplot_python_venv\Scripts\python -m pip install --only-binary=:all: -e "waveplot_web[dev,server]" -c waveplot_web\constraints-py313.txt
waveplot_python_venv\Scripts\waveplot serve Waveplot_exe
```

Then open <http://localhost:8050/>. See [`waveplot_web/README.md`](waveplot_web/README.md) for the
command-line tools, the Python API and tests, and `waveplot_web/docs/dev/` for the file formats,
architecture and the list of v407 behaviours.

## Problems found in the original Pascal (WavePlot v407)

Found while porting `Turbo pascal version/` (file:line references below). In WavePlot Web the
**Standard** numerics mode uses corrected behaviour; the **v407** mode reproduces the numerical
behaviours marked ↺ exactly (so old results can be reproduced), and a badge on each page lists the
ones that affected the current view. Items without ↺ are fixed in both modes.

### Scientific / numerical errors

| | Problem | Where |
|---|---|---|
| ↺ | **Excess shear stress (ESS) uses the wrong sign** in the shear term (+½(s11−s22)·sin2θ instead of −½(s11−s22)·sin2θ), so the shear and normal stresses are taken on *different* planes for any rotation angle θ other than 0° or 90°. | `DATA_RSC.PAS:1171-1175` |
| | **Principal-stress direction is stored mirrored**: `Ang = ½·Arctan0(2·s12, −(s11−s22))` is the reflection of the most compressive direction; v407 only looked right because the bar was drawn at −Ang on a flipped screen axis. The port computes the physical direction (verified to be an eigenvector). | `DATA_RSC.PAS:1160` |
| ↺ | **Zero-phase Butterworth filter (`&f`) is not zero phase**: the backward pass continues from the forward pass's internal filter state instead of starting fresh (not equivalent to forward–backward filtering). | `FILTERS.PAS:90-110` |
| ↺ | **FFT filters and Q attenuation modify the wrong mirror bin** (N−1−i instead of N−i), so the filtered spectrum is not Hermitian; only the real part is kept afterwards. | `FOURIER.PAS:182-224`, `DSPFORM.PAS:245-253` |
| ↺ | **Acceleration particle-motion trajectories have the wrong sign**: (v_prev − v)/dt. | `DATA_RSC.PAS:1698` |
| ↺ | Spectral **taper indices are off by one sample** (a 1-based sample number is used as a 0-based offset). | `DSPFORM.PAS:302-324` |
| ↺ | **Empty spectral bins report a 270° phase** because `Arctan0(0,0) = −π/2`; phase is also forced to increase monotonically. The forward FFT uses e^{+i} (phase sign opposite to the usual convention). | `FUNCT.PAS:113-130`, `DSPFORM.PAS:350-361`, `FOURIER.PAS:118-165` |
| ↺ | **Cosine colour-level distribution overshoots**: ½·(1−cos πx)⁴ reaches 8× the data range, so most levels lie above the maximum. | `dsp/DSPSNAP.PAS:97-98` |
| ↺ | **Window/extract (`J`, `&W`) with 5 parameters reads the decimation factor from the taper parameter** (`Params[4]` instead of `Params[5]`). | `DSPFORM.PAS:173` |
| ↺ | **Decimation (`&D`) has no anti-alias filter**, drops the first sample (keeps samples q, 2q, …) without shifting the start time, and sets the end time one sample too late. | `DSPFORM.PAS:1084-1098` |
| ↺ | **Signed power (`E`) loses the sign for p = 1/3** (and similar exponents): the "integer exponent" test `round(p) = trunc(p)` is true whenever frac(p) < 0.5. | `FUNCT.PAS:393-419` |
| ↺ | **Rotation towards a source (`&R`) uses grid indices instead of distances**, which is wrong when dx, dy and dz differ. | `DSPFORM.PAS:1171-1172` |
| ↺ | **Binary formula operations (`+ − * / C R r`) work sample by sample and take the *right* operand's time axis and length** (global `Hmap`), so traces with different sampling are combined incorrectly. | `DSPFORM.PAS:856-906` |
| | **Uninitialised buffers**: the `$g/$h/$*` operators and binary operators without a following `[n]` read memory that was never set. | `DSPFORM.PAS:856-857, 952-953` |
| | Integration (`I`) uses the left rectangle rule and differentiation (`D`) a forward difference (half-sample shift) — low-order schemes, reproduced in v407 mode. | `DSPFORM.PAS:753-782` |

### Formula language

* There is **no operator precedence**: a binary operator takes the *whole rest* of the formula as
  its right operand, so `[1]*[2]+[3]` means `[1]*([2]+[3])` (kept for legacy formulas; the
  modern syntax uses normal precedence).
* Unknown characters are silently skipped, unbalanced brackets read past the end of the string,
  and a parameter followed by `E` (e.g. `M,2E`) is not parsed (the number scanner accepts `e/E`).
  The port reports all of these as diagnostics.
* `L,hi,lo` loops forever when hi ≤ lo; `$H` does nothing but consumes the rest of the formula.
* The phase/group-velocity functions `V W Z A B` do nothing: their code is commented out.

### Data handling and limits

* **Capacity limits overflow silently**: beyond 250 snapshots, 510 histories or 50 dumps the
  last slot is overwritten again and again (the `cg11` run has 61 dumps, so 11 were lost);
  histories are truncated to 4096 samples; at most 8 traces, a 101×101 display grid, 24 contour
  lines and 6×6 panels.
* Stale loop indices in `DI_ReadGeom`: `GeomMap^[i]` instead of `[G]`, and memory allocated for
  `Stopes^[i].Outline` instead of `[ii]` (`DATA_RSC.PAS:1463-1464, 1569`); the `.OTL` outline file
  is re-read for every stope.
* `.ASC` export writes every trace with trace 1's time axis and length (`DATA_RSC.PAS:1790-1804`).
* Snapshot decimation for display uses trailing (asymmetric) block averages
  (`DATA_RSC.PAS:850-900`).

### Display, plotting and user interface

* `|x|` and log-scale seismograms are drawn **one sample shifted** (the sample index is not
  advanced after `MoveTo`, unlike the linear plot) (`dsp/DSPSEISM.PAS:488, 537`).
* HP-GL plotter: the vertical offset uses the window *width* (`plot.pas:90`); log–log plots read
  an uninitialised sample index (`plt/PLSEISM.PAS:228`); colour maps cannot be plotted at all.
* The `D` key only cycles 3 of the 7 level distributions, the "SqLog" distribution is not
  implemented (stale levels are used), and `<` can drive the number of levels to 0 (division by
  zero) (`WVPLOT.PAS:454-458`, `dsp/DSPSNAP.PAS:87-106`).
* Help pages are placeholders (every context shows the geometry page); several on-screen hints
  name the wrong keys; Esc does not leave the main menu; any key press during drawing leaves a
  partly drawn plot; screen BMP export is 16-colour only (corrupt in 256-colour modes).

The complete catalogue, including how each item is handled, is in
[`waveplot_web/docs/dev/compat.md`](waveplot_web/docs/dev/compat.md). (Separately, the two older
Python `.map` readers, `utilities/read_map_file.py` and `Turbo pascal version/read_map.py`, use
wrong record sizes and fail on the sample `.map` file.)

## Earlier Tkinter GUI (`waveplot_python/`)

An earlier Python port, a Tkinter + Matplotlib desktop GUI. WavePlot is a Python-based waveform plotting and analysis tool for WAVE simulation outputs, originally ported from Turbo Pascal. The current GUI implementation lives in the `waveplot_python` directory and pairs Tkinter with Matplotlib to parse `.map` files along with their corresponding history, snapshot, and dump files.

### Installation

#### Prerequisites

- Python 3.7 or higher
- pip
- The dependencies listed in `requirements.txt` (`tkinter` ships with Python on Windows/macOS; Linux distributions may package it separately)

#### Quick setup

```
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### Launch the Tkinter GUI

Run the new interface from the repository root:

```
python -m waveplot_python.main [path\to\project.map]
```

Passing a `.map` path preloads the project; omit it to open files later via `File -> Open .map...`. You can also launch with `python waveplot_python/main.py`.

### Additional documentation

I created `waveplot_python/software_architecture.md` for more context regarding the data flow.

## Legacy assets

Older references remain in the repository:

- `Turbo pascal version/` contains the original Pascal source.
