"""Synthetic recordings for tests and offline demos.

The generator produces a plausible N-back-like stream: three load levels with
different frontal-theta / posterior-alpha mixes, slow drift, broadband noise
and a weak 50 Hz line component. It is not a physiological model; it only
needs to be separable after preprocessing so the training loop can be tested
without recorded data.
"""
from __future__ import annotations

import numpy as np

from hotb_2026.brainaccess_connection import MONTAGE

from .schema import Recording, StateChange

# Relative band amplitudes per class: (frontal theta, posterior alpha).
_CLASS_MIX = {
    0: (6.0, 1.0),
    1: (4.0, 4.0),
    2: (2.0, 8.0),
}


def make_synthetic_recording(
    duration_s: float = 600.0,
    sfreq: float = 250.0,
    electrodes: tuple[str, ...] | None = None,
    state_duration_s: float = 20.0,
    seed: int = 0,
    session_id: str = "synthetic-1",
    subject_id: str = "synthetic",
) -> Recording:
    """Return a synthetic :class:`Recording` with cycling 0/1/2 states."""
    if duration_s <= 0 or sfreq <= 0 or state_duration_s <= 0:
        raise ValueError("duration_s, sfreq and state_duration_s must be positive")

    electrodes = tuple(electrodes or MONTAGE.names)
    n_channels = len(electrodes)
    n_samples = int(round(duration_s * sfreq))
    if n_samples < 2:
        raise ValueError("duration_s is too short for the requested sfreq")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    signal = rng.normal(0.0, 5.0, size=(n_channels, n_samples))
    signal += 10.0 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))
    signal += 1.5 * np.sin(2 * np.pi * 50.0 * t)

    n_states = int(np.ceil(duration_s / state_duration_s))
    changes: list[StateChange] = []
    for index in range(n_states):
        onset = index * state_duration_s
        label = index % 3
        changes.append(StateChange(time_s=float(onset), label=label))

        end = min((index + 1) * state_duration_s, duration_s)
        mask = (t >= onset) & (t < end)
        if not np.any(mask):
            continue

        theta_amp, alpha_amp = _CLASS_MIX[label]
        # Smooth the envelope so state changes are not perfect clicks.
        envelope = np.ones(mask.sum())
        ramp = min(int(0.25 * sfreq), envelope.size // 2)
        if ramp > 0:
            envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
            envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)

        for channel, name in enumerate(electrodes):
            phase = rng.uniform(0, 2 * np.pi)
            if name in {"F3", "F4"}:
                signal[channel, mask] += theta_amp * envelope * np.sin(
                    2 * np.pi * 6.0 * t[mask] + phase
                )
            elif name in {"P3", "P4", "O1", "O2"}:
                signal[channel, mask] += alpha_amp * envelope * np.sin(
                    2 * np.pi * 10.0 * t[mask] + phase
                )

    return Recording(
        session_id=session_id,
        subject_id=subject_id,
        electrodes=electrodes,
        signal=signal,
        time_s=t,
        state_changes=tuple(changes),
        sfreq=float(sfreq),
    )


def write_synthetic_session(path: str, **kwargs) -> Recording:
    """Generate one synthetic recording and write it in the raw session format."""
    recording = make_synthetic_recording(**kwargs)
    recording.to_dir(path)
    return recording


__all__ = ["make_synthetic_recording", "write_synthetic_session"]
