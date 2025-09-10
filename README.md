# WavePlot

A Python-based waveform plotting and analysis tool for scientific simulation data, originally ported from Turbo Pascal. WavePlot provides a GUI interface for visualizing waveform data from history files (.hst), model files (.m), and map files (.map).

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages (install via pip):

```bash
pip install numpy pandas matplotlib scipy tkinter
```

### Setup

1. Clone the repository:
```bash
git clone https://github.austin.utexas.edu/texnet/WavePlot.git
cd WavePlot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### WaveViewer GUI Application

Launch the main GUI application:

```bash
python Manolis_pythhon_version/WaveViewer_stable.py
```

#### Command Line Options

- `--debug`: Enable debug logging mode for detailed troubleshooting (automatically logs to a 'logs' directory)

```bash
python Manolis_pythhon_version/WaveViewer_stable.py --debug
```


### Map File Reader Utility

The `read_map_file.py` utility provides command-line access to .map file analysis:

#### Basic Usage

```bash
# Print summary to console
python python_code/read_map_file.py --map-file path/to/your/file.map

# Save summary to file
python python_code/read_map_file.py --map-file path/to/your/file.map -o summary.txt
```

#### Example Commands

```bash
# Quick check of .map file contents
python python_code/read_map_file.py --map-file Waveplot_exe/mt_slip5.map

# Save summary to a text file
python python_code/read_map_file.py --map-file Waveplot_exe/mt_slip5.map -o mt_slip5_analysis.txt
```

