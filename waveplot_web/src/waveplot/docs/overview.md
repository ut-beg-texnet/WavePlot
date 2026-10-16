# WavePlot

WavePlot visualises the output of **WAVE** simulations (wave propagation and rock mechanics).
It is a modern, browser-based successor to WavePlot v407 (Turbo Pascal).

## Workflow

1. **Datasets** — pick a run (a `.map` file and its companions `.snp .hst .dmp .geo …`).
   The run card summarises the grid, timestep, materials (with P- and S-wave speeds) and any
   problems found in the files.
2. **Snapshots** — 2-D planes of a field over time: colour maps, contours or multi-line ("wiggle")
   views, geometry overlays, animation, multi-panel grids, and a probe that plots the time series
   of any cell. Click a receiver marker to open its histories.
3. **Dumps** — the full stress tensor and velocity on a plane: principal-stress crosses, velocity
   arrows, maps of excess shear stress (ESS), maximum shear, principal / mean / von Mises
   stresses, differences between two times, and Mohr circles for any point.
4. **Histories** — receiver time series with processing pipelines (integrate, filter, spectra,
   τp, rotations, …), written in a modern expression syntax or in the original v407 formula
   language. Tabs give spectra, spectrograms, response spectra, a record section and metrics.
5. **Geometry** — the model in 3-D (materials, sources, faults, receivers, snapshot planes),
   cross-sections, particle motion and crack dumps.
6. **Compare** — the same quantity from two runs side by side or as a difference.

## Numerics: Standard vs v407

The switch in the header selects how calculations are done:

* **Standard** — conventional, corrected numerics (e.g. trapezoidal integration, zero-phase
  filtering with SciPy, consistent stress rotations).
* **v407** — reproduces the original program's calculations, including its known quirks, so
  that old figures and numbers can be reproduced exactly. A badge lists the quirks that affected
  the current view. See *v407 differences*.

## Units and axes

Values carry SI units (Pa, m/s, m, s). Grid cell *i* spans ((i−1)·dx, i·dx); values are plotted at
cell centres, shifted by half a cell for staggered variables (e.g. velocities, shear stresses).
In planes the first record axis is horizontal; a vertical *j* axis points upwards, as in v407.

## Command line and Python

```
waveplot serve <folder or .map> [--port 8050] [--prod]
waveplot info run.map
waveplot convert run.map --to netcdf
waveplot formula run.map "[2]I&f,1,10" --mode v407 --explain
waveplot export run.map --series "S11|ik|j=50" -o s11.mp4
```

```python
import waveplot, xarray as xr
ds = waveplot.open_dataset("run.map")
da = ds.series["S11|ik|j=50"].to_xarray()          # (time, x, z) with coordinates in metres
tree = xr.open_datatree("run.map", engine="wave")   # lazy access to everything
```
