"""WavePlot — interactive visualisation of WAVE simulation outputs.

Python port and modernisation of the Turbo Pascal WavePlot v407.

>>> import waveplot
>>> ds = waveplot.open_dataset("Waveplot_exe/mt_slip5.map")
"""

from typing import Literal

from .model.dataset import WaveDataset, open_dataset

Mode = Literal["standard", "v407"]

__version__ = "0.1.0"
__all__ = ["Mode", "WaveDataset", "__version__", "open_dataset"]
