"""Parameters of the per-window EEG preprocessing pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass


def _require_finite(name: str, value: float) -> None:
    try:
        finite = math.isfinite(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not finite:
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings for :class:`~hotb_2026.preprocessing.Preprocessor`.

    The input rate is not defined here by accident: this package has no
    dependency on ``brainaccess_connection``. Pass ``MONTAGE.sfreq`` when
    building the config so the montage remains the single source of truth.

    Set a stage to ``None`` to skip it (for example while testing one filter in
    isolation or while plotting step-by-step results). Parameters that belong
    to a disabled stage are still validated: an invalid value is a mistake even
    if the stage is currently off.
    """

    sfreq: float = 250.0
    notch_freq: float | None = 50.0
    notch_q: float = 30.0
    bandpass: tuple[float, float] | None = (0.5, 45.0)
    filter_order: int = 4
    target_sfreq: float | None = 200.0
    zscore: bool = True
    clip: float | None = 15.0

    def __post_init__(self) -> None:
        _require_finite("sfreq", self.sfreq)
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")

        _require_finite("notch_q", self.notch_q)
        if self.notch_q <= 0:
            raise ValueError(f"notch_q must be positive, got {self.notch_q}")

        _require_finite("filter_order", self.filter_order)
        if self.filter_order < 1:
            raise ValueError(f"filter_order must be positive, got {self.filter_order}")
        # ``butter`` accepts 4 and 4.0 but rejects 4.5; reject it here so the
        # error points at the config instead of at SciPy.
        if float(self.filter_order) != int(self.filter_order):
            raise ValueError(
                "filter_order must be a whole number, "
                f"got {self.filter_order}"
            )

        nyquist = self.sfreq / 2

        if self.notch_freq is not None:
            _require_finite("notch_freq", self.notch_freq)
            if not 0 < self.notch_freq < nyquist:
                raise ValueError(
                    f"notch_freq must be in (0, {nyquist}) Hz, "
                    f"got {self.notch_freq}"
                )

        if self.bandpass is not None:
            if len(self.bandpass) != 2:
                raise ValueError(
                    "bandpass must contain exactly two edges (low, high), "
                    f"got {self.bandpass}"
                )
            low, high = self.bandpass
            _require_finite("bandpass low edge", low)
            _require_finite("bandpass high edge", high)
            if not 0 < low < high:
                raise ValueError(
                    "bandpass must be (low, high) with 0 < low < high, "
                    f"got {self.bandpass}"
                )
            if high >= nyquist:
                raise ValueError(
                    f"bandpass high edge must be below Nyquist ({nyquist} Hz), "
                    f"got {high}"
                )

        if self.target_sfreq is not None:
            _require_finite("target_sfreq", self.target_sfreq)
            if self.target_sfreq <= 0:
                raise ValueError(
                    f"target_sfreq must be positive, got {self.target_sfreq}"
                )
            if (
                self.bandpass is not None
                and self.bandpass[1] >= self.target_sfreq / 2
            ):
                raise ValueError(
                    "bandpass high edge must be below the target Nyquist "
                    f"({self.target_sfreq / 2} Hz), got {self.bandpass[1]}"
                )

        if self.clip is not None:
            _require_finite("clip", self.clip)
            if self.clip <= 0:
                raise ValueError(f"clip must be positive, got {self.clip}")
