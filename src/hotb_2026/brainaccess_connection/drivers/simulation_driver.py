"""Fake headset used when no BrainAccess device is available."""
from __future__ import annotations

import time

import numpy as np

from ..driver import DeviceError
from ..models.montage import Montage


class SimulatedDriver:
    """Fake headset: 10 Hz alpha on every channel plus noise, paced in real time."""

    def __init__(self, montage: Montage, chunk_ms: float = 40.0, seed: int = 0):
        self.montage = montage
        self.sfreq = montage.sfreq
        self._n_chunk = max(1, round(montage.sfreq * chunk_ms / 1000))
        self._chunk_s = self._n_chunk / montage.sfreq
        self._rng = np.random.default_rng(seed)
        self._amp = np.full(len(montage.electrodes), 20.0, dtype=np.float32)
        self._sample = 0
        self._running = False

    def open(self) -> None:
        pass

    def start(self) -> None:
        self._sample = 0
        self._running = True

    def read(self) -> np.ndarray:
        if not self._running:
            raise DeviceError("Simulator not started")
        time.sleep(self._chunk_s)
        t = (self._sample + np.arange(self._n_chunk)) / self.sfreq
        self._sample += self._n_chunk
        alpha = np.sin(2 * np.pi * 10 * t)[:, None] * self._amp[None, :]
        noise = self._rng.normal(0, 5, size=(self._n_chunk, len(self._amp)))
        return (alpha + noise).astype(np.float32)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        pass
