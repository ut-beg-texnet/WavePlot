# WavePlot

WavePlot is a Python-based waveform plotting and analysis tool for WAVE simulation outputs, originally ported from Turbo Pascal. The current GUI implementation lives in the `waveplot_python` directory and pairs Tkinter with Matplotlib to parse `.map` files along with their corresponding history, snapshot, and dump files.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip
- The dependencies listed in `requirements.txt` (`tkinter` ships with Python on Windows/macOS; Linux distributions may package it separately)

### Quick setup

```
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

## Launch the WavePlot GUI

Run the new interface from the repository root:

```
python -m waveplot_python.main [path\to\project.map]
```

Passing a `.map` path preloads the project; omit it to open files later via `File -> Open .map...`. You can also launch with `python waveplot_python/main.py`.

## Additional documentation

I created `waveplot_python/software_architecture.md` for more context regarding the data flow.

## Legacy assets

Older references remain in the repository:

- `Turbo pascal version/` contains the original Pascal source.
