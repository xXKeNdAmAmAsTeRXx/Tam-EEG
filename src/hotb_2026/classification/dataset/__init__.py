"""Training dataset construction for the REVE + SVM classifier."""
from .builder import (
    DEFAULT_LABEL_NAMES,
    Window,
    build_dataset,
    iter_windows,
    label_at,
    load_recordings,
)
from .schema import (
    FORMAT_VERSION,
    EegDataset,
    Recording,
    StateChange,
    normalize_state_label,
)
from .synthetic import make_synthetic_recording, write_synthetic_session

__all__ = [
    "DEFAULT_LABEL_NAMES",
    "FORMAT_VERSION",
    "EegDataset",
    "Recording",
    "StateChange",
    "Window",
    "build_dataset",
    "iter_windows",
    "label_at",
    "load_recordings",
    "make_synthetic_recording",
    "normalize_state_label",
    "write_synthetic_session",
]
