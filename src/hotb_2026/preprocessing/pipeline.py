"""Notch -> band-pass -> resample -> z-score -> clip, one window at a time."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos

from .config import PreprocessingConfig

_STD_EPS = 1e-12


def _sos_padlen(sos: np.ndarray) -> int:
    """Default ``sosfiltfilt`` padding for ``sos`` sections.

    Mirrors ``scipy.signal._signaltools._validate_pad`` so that a too-short
    window raises a clear error before SciPy sees it.
    """
    ntaps = 2 * len(sos) + 1 - min(
        int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())
    )
    return 3 * ntaps


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    centered = x - mean
    # A flat channel carries no information; return zeros instead of NaNs.
    return np.divide(
        centered,
        std,
        out=np.zeros_like(centered),
        where=std > _STD_EPS,
    )


class Preprocessor:
    """Transforms one EEG window into a clean, standardized window.

    Input is ``(n_channels, n_samples)`` in montage order; output keeps the
    channel order and has approximately ``n_samples * output_sfreq / sfreq``
    samples (``resample_poly`` rounds up when the ratio is not exact). Filters
    are designed once in ``__init__``, so calls are cheap enough for a streamer
    callback. The input array is never modified.

    The pipeline runs exactly in this order: notch, band-pass, resample,
    z-score, clip.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        cfg = self.config

        self._notch_sos: np.ndarray | None = None
        self._notch_padlen = 0
        if cfg.notch_freq is not None:
            b, a = iirnotch(cfg.notch_freq, cfg.notch_q, fs=cfg.sfreq)
            self._notch_sos = tf2sos(b, a)
            self._notch_padlen = _sos_padlen(self._notch_sos)

        self._bandpass_sos: np.ndarray | None = None
        self._bandpass_padlen = 0
        if cfg.bandpass is not None:
            low, high = cfg.bandpass
            self._bandpass_sos = butter(
                cfg.filter_order,
                [low, high],
                btype="bandpass",
                fs=cfg.sfreq,
                output="sos",
            )
            self._bandpass_padlen = _sos_padlen(self._bandpass_sos)

        self._min_samples = max(self._notch_padlen, self._bandpass_padlen) + 1

        self._up: int | None = None
        self._down: int | None = None
        if cfg.target_sfreq is not None and not math.isclose(
            cfg.target_sfreq, cfg.sfreq
        ):
            ratio = Fraction(cfg.target_sfreq / cfg.sfreq).limit_denominator(1000)
            self._up = ratio.numerator
            self._down = ratio.denominator

    @property
    def min_samples(self) -> int:
        """Shortest window accepted by :meth:`process` (samples)."""
        return self._min_samples

    @property
    def output_sfreq(self) -> float:
        """Sample rate after resampling (equal to ``sfreq`` if skipped)."""
        # ``_up`` is ``None`` when there is no resampling stage, including the
        # case where ``target_sfreq`` only differs from ``sfreq`` by rounding.
        # Report the rate that ``process`` actually produces, not the requested
        # target in that case.
        target = self.config.target_sfreq
        if self._up is None or target is None:
            return self.config.sfreq
        return target

    def process(self, window: np.ndarray) -> np.ndarray:
        """Preprocess one window and return it as ``float32``.

        Parameters
        ----------
        window:
            ``(n_channels, n_samples)`` array; columns are time samples.

        Raises
        ------
        ValueError
            If the window is not 2-D, has no channels, contains non-finite
            values, or is shorter than :attr:`min_samples`.
        """
        x = np.asarray(window)
        if x.ndim != 2:
            raise ValueError(
                "window must be 2-D (n_channels, n_samples), "
                f"got shape {x.shape}"
            )
        if x.shape[0] < 1:
            raise ValueError("window must have at least one channel")
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError(
                "window must contain real numeric values, "
                f"got dtype {x.dtype}"
            )
        if x.shape[1] < self._min_samples:
            raise ValueError(
                f"window too short: need at least {self._min_samples} samples "
                f"at {self.config.sfreq:g} Hz, got {x.shape[1]}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("window contains non-finite values (NaN or Inf)")

        x = np.array(x, dtype=np.float64, copy=True)

        if self._notch_sos is not None:
            # Pass padlen explicitly so ``min_samples`` and SciPy cannot drift
            # apart if the SciPy default ever changes.
            x = sosfiltfilt(
                self._notch_sos, x, axis=-1, padlen=self._notch_padlen
            )
        if self._bandpass_sos is not None:
            x = sosfiltfilt(
                self._bandpass_sos, x, axis=-1, padlen=self._bandpass_padlen
            )
        if self._up is not None and self._down is not None:
            # "line" padding keeps the edge transient of an isolated window
            # smaller than zero padding would.
            x = resample_poly(x, self._up, self._down, axis=-1, padtype="line")
        if self.config.zscore:
            x = _zscore(x)
        if self.config.clip is not None:
            np.clip(x, -self.config.clip, self.config.clip, out=x)

        return x.astype(np.float32, copy=False)
