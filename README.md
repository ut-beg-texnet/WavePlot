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

#### GUI Features

1. **Open .hst File**: Load history files containing waveform data
2. **Plot All Waveforms**: Display all available waveforms in a single window
3. **Select Waveforms**: Choose specific waveforms to plot
4. **Plot Model Geometry**: Visualize 3D model structure
5. **Export to CSV**: Save waveform data for external analysis
6. **Open .map File**: Load map files and automatically resolve related data files

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
# Analyze a map file in the Waveplot_exe directory
python python_code/read_map_file.py --map-file Waveplot_exe/mt_slip5.map

# Save detailed analysis to a text file
python python_code/read_map_file.py --map-file Waveplot_exe/mt_slip5.map -o mt_slip5_analysis.txt

# Quick check of map file contents
python python_code/read_map_file.py --map-file data/simulation.map
```

#### Output Example

```
.MAP summary:
  Snapshots : 5
  Histories : 12
  Dumps     : 3
  Geometries: 1
  CrackData : 0

History timing (from first history record):
  Samples   : 1001
  Range     : 0.0 .. 0.36
  Time step : 0.00036

Grid (from first geometry record):
  Dimensions: I=50  J=30  K=20
  Spacing   : dx=0.1  dy=0.1  dz=0.1
```

## File Structure

```
WavePlot/
├── Manolis_pythhon_version/          # Main Python application
│   ├── WaveViewer_stable.py         # Main GUI application
│   ├── mFile.py                     # Model file utilities
│   └── logs/                        # Application logs
├── python_code/                     # Utility modules
│   └── read_map_file.py            # Map file reader
├── Turbo pascal version/            # Original Pascal source code
├── Waveplot_exe/                   # Example data files
└── waveplot_venv/                  # Python virtual environment
```

## Supported File Formats

### History Files (.hst)
- Binary format containing time-series waveform data
- Stores velocity histories, stress data, and other simulation outputs
- Automatically parsed with time axis generation

### Model Files (.m)
- Text format containing simulation parameters
- Includes timestep (dt), cycles, and history variable definitions
- Used to configure plotting and analysis parameters

### Map Files (.map)
- Binary metadata format organizing simulation outputs
- Contains pointers to snapshots, histories, dumps, and geometry data
- Enables automatic file discovery and organization

## Development

### Logging

The application includes comprehensive logging:
- Log files are stored in `Manolis_pythhon_version/logs/`
- Timestamped log files for each session
- Debug mode available via `--debug` flag
- Separate log levels for different components

### Error Handling

- Graceful handling of missing files
- User-friendly error messages
- Automatic fallback to default parameters when needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Originally developed in Turbo Pascal
- Ported to Python for modern cross-platform compatibility
- Maintains compatibility with original data formats
