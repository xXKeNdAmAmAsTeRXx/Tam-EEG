"""Project defaults for the BrainAccess MINI headset.

Edit this file, not the library modules, to change the headset or montage.
The CLI reads its defaults from here, so a plain ``uv run hotb-brainaccess``
uses the values below.
"""
from __future__ import annotations

from .models.montage import Electrode, Montage

DEVICE_NAME = "BA EXG8 005"

# 250 Hz is supported by the MINI hardware. Pass ``--sfreq 500`` for the
# higher rate; simulation accepts any positive rate.
MONTAGE = Montage(
    electrodes=(
        Electrode("Fp2", pin=0),
        Electrode("P4", pin=1),
        Electrode("F4", pin=2),
        Electrode("F3", pin=3),
        Electrode("PZ", pin=4),
        Electrode("P3", pin=5),
        Electrode("Fp1", pin=6),
        Electrode("Pz", pin=7),
    ),
    sfreq=250.0,
    bias_electrode=7,  # 0-based electrode index, or None to disable feedback
)
