"""Result of one live classification."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """One classified EEG window on the stream timeline.

    Parameters
    ----------
    label:
        Predicted class id, ``0..n_classes - 1``.
    label_name:
        Human-readable name of the predicted class from the bundle metadata.
    confidence:
        Probability of the predicted class, equal to ``max(probabilities)``.
    probabilities:
        Class name -> probability, in the class order recorded in the bundle.
    current_time:
        Stream time at which the window ended, in seconds.
    state_change_time:
        Stream time at which this predicted label was first reported. Live
        inference has no ground truth, so this is the start of the current
        predicted state, not the timestamp of a recorded state change.
    window_start:
        ``current_time - window_s``.
    window_end:
        Equal to ``current_time``.
    model_version:
        ``model_version`` from ``metadata.json``.
    """

    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    current_time: float
    state_change_time: float
    window_start: float
    window_end: float
    model_version: str


__all__ = ["Prediction"]
