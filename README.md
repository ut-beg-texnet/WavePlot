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
