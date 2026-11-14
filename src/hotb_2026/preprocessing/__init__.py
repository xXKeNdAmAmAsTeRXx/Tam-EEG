"""Per-window EEG preprocessing built on top of ``brainaccess_connection``.

The package does not connect to hardware and does not own a buffer: it takes
the rolling window already produced by ``EEGStreamer`` and returns a clean,
standardized window for downstream processing.
"""
from .config import PreprocessingConfig
from .pipeline import Preprocessor
from .windowing import WindowCallback, WindowedPreprocessor

__all__ = [
    "PreprocessingConfig",
    "Preprocessor",
    "WindowCallback",
    "WindowedPreprocessor",
]
