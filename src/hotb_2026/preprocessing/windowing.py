"""Turn the streamer's rolling window into one preprocessing call per window."""
from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np

from .pipeline import Preprocessor

log = logging.getLogger(__name__)

WindowCallback = Callable[[np.ndarray, float], None]


class WindowedPreprocessor:
    """``EEGStreamer`` processor that preprocesses every completed window.

    ``EEGStreamer`` calls processors after every chunk with the whole rolling
    window. This adapter waits until the stream time crosses a multiple of
    ``window_s`` and then preprocesses the completed window. It therefore emits
    one result per window and reuses the streamer's ring buffer instead of
    keeping a second copy of the signal.

    Parameters
    ----------
    preprocessor:
        Configured :class:`~hotb_2026.preprocessing.Preprocessor`.
    sfreq:
        Input sample rate, normally ``MONTAGE.sfreq``.
    window_s:
        Window length in seconds. Pass ``streamer.window_s`` to preprocess
        exactly the whole established streamer window. It must not exceed the
        streamer's window length, otherwise the adapter cannot recover a full
        window.
    on_window:
        Called as ``on_window(processed, t)`` on the streamer thread, where
        ``processed`` is ``(n_channels, n_samples * output_sfreq / sfreq)`` and
        ``t`` is the current stream time in seconds.

    Alignment
    ---------
    A chunk usually does not end exactly on a window boundary. When the
    streamer's rolling window is longer than ``window_s``, the adapter slices
    the exact completed window out of the buffer, so windows stay aligned and
    do not overlap. When both windows are the same length, the samples just
    before the boundary are no longer available; the adapter then falls back to
    the latest full window and counts it in :attr:`shifted_windows`. Choose a
    chunk size that divides ``window_s * sfreq`` (for example 25 samples at
    250 Hz for a 4 s window) to avoid the fallback entirely.

    If one chunk crosses several window boundaries, only the latest completed
    window is processed; the others are counted in :attr:`skipped_windows` and
    logged at DEBUG level. This is a consequence of receiving the rolling
    window rather than raw chunks.

    :meth:`reset` is called automatically when stream time moves backwards,
    which happens when the same streamer is restarted.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        *,
        sfreq: float,
        window_s: float,
        on_window: WindowCallback,
    ):
        if not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"sfreq must be a positive finite number, got {sfreq}")
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError(
                f"window_s must be a positive finite number, got {window_s}"
            )
        self.preprocessor = preprocessor
        self.sfreq = float(sfreq)
        self.window_s = float(window_s)
        self._window_samples = max(1, round(self.sfreq * self.window_s))
        self._on_window = on_window
        self.reset()

    @property
    def window_samples(self) -> int:
        return self._window_samples

    @property
    def skipped_windows(self) -> int:
        """Windows dropped because one chunk crossed several boundaries."""
        return self._skipped_windows

    @property
    def shifted_windows(self) -> int:
        """Windows processed from the latest full window instead of the boundary.

        This happens when the streamer window is not longer than the
        preprocessing window and a chunk overshoots the boundary. The result is
        shifted by ``offset`` samples and can overlap the next window.
        """
        return self._shifted_windows

    def reset(self) -> None:
        """Forget window boundaries and counters (e.g. before restarting)."""
        self._next_end = self._window_samples
        self._last_end = 0
        self._skipped_windows = 0
        self._shifted_windows = 0

    def __call__(self, window: np.ndarray, t: float) -> None:
        """Process one completed window. Matches ``EEGStreamer.Processor``."""
        if not math.isfinite(t):
            raise ValueError(f"t must be a finite stream time, got {t}")
        end = int(round(t * self.sfreq))
        if end < self._last_end:
            log.debug(
                "Stream time moved backwards (%.3f s); resetting windows", t
            )
            self.reset()
        self._last_end = end

        if end < self._next_end:
            return

        data = np.asarray(window)
        if data.ndim != 2:
            raise RuntimeError(
                "window must be 2-D (n_channels, n_samples), "
                f"got shape {data.shape}"
            )
        if data.shape[1] < self._window_samples:
            raise RuntimeError(
                "The streamer window is shorter than the preprocessing window: "
                f"got {data.shape[1]} samples, need {self._window_samples}. "
                "Pass a larger window_s to EEGStreamer."
            )

        crossed = (end - self._next_end) // self._window_samples + 1
        if crossed > 1:
            self._skipped_windows += crossed - 1
            log.debug(
                "Skipped %d preprocessing window(s); a chunk crossed several "
                "boundaries",
                crossed - 1,
            )

        latest_boundary = self._next_end + (crossed - 1) * self._window_samples
        offset = end - latest_boundary

        if offset == 0 or data.shape[1] >= self._window_samples + offset:
            # Exact completed window: [latest_boundary - N, latest_boundary).
            stop = data.shape[1] - offset
            start = stop - self._window_samples
            segment = data[:, start:stop]
        else:
            # Equal streamer/preprocessing windows and a chunk that overshoots:
            # the first samples of the completed window are already overwritten.
            self._shifted_windows += 1
            log.debug(
                "Streamer window too small to align to the boundary; using the "
                "latest full window (shifted by %d samples)",
                offset,
            )
            segment = data[:, -self._window_samples :]

        result = self.preprocessor.process(segment)
        self._next_end = latest_boundary + self._window_samples
        self._on_window(result, t)
