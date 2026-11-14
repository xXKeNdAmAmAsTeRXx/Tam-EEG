"""Project defaults for the BrainAccess MINI headset.

Edit this file, not the library modules, to change the headset or montage.
The CLI reads its defaults from here, so a plain ``uv run hotb-brainaccess``
uses the values below.
"""
from __future__ import annotations

from .models.montage import Electrode, Montage

DEVICE_NAME = "BA MINI 078"

# 200 Hz is a simulation-only default: the MINI hardware streams at 250 or
# 500 Hz only. Pass ``--sfreq 250`` (or change this value) when using a headset.
MONTAGE = Montage(
    electrodes=(
        Electrode("F3", pin=0),
        Electrode("F4", pin=1),
        Electrode("C3", pin=2),
        Electrode("C4", pin=3),
        Electrode("P3", pin=4),
        Electrode("P4", pin=5),
        Electrode("O1", pin=6),
        Electrode("O2", pin=7),
    ),
    sfreq=250.0,
    bias_electrode=7,  # 0-based electrode index, or None to disable feedback
)
