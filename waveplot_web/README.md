# WavePlot Web

Interactive, browser-based visualisation of WAVE simulation outputs — a Python port and
modernisation of the Turbo Pascal WavePlot v407 (see `../Turbo pascal version/`).

*(Full documentation is added as the app grows; see `docs/`.)*

```
py -3.13 -m venv ..\waveplot_python_venv
..\waveplot_python_venv\Scripts\python -m pip install --only-binary=:all: -e ".[dev,server]" -c constraints-py313.txt
..\waveplot_python_venv\Scripts\waveplot info ..\Waveplot_exe\mt_slip5.map
```
