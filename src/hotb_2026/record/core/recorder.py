"""Background EEG recorder used while a PsychoPy procedure runs."""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np


class EEGRecorder:
    """Continuously reads a driver and keeps samples for active states.

    Samples collected while the current state is ``None`` are discarded
    ("cut out"), but the global sample counter still advances so that stream
    time stays continuous.
    """

    def __init__(
        self,
        driver,
        montage,
        *,
        poll_interval: float = 0.005,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.driver = driver
        self.montage = montage
        self.sfreq = float(montage.sfreq)
        self.poll_interval = poll_interval
        self._sleep = sleep

        self._lock = threading.Lock()
        self._sample_count = 0
        self._current_label: str | None = None
        self._state_changes: list[tuple[float, str]] = []
        self._signal_chunks: list[np.ndarray] = []
        self._time_chunks: list[np.ndarray] = []

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._started:
            raise RuntimeError("EEGRecorder is already started")
        self.driver.open()
        self.driver.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="EEGRecorder", daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        try:
            self.driver.stop()
        finally:
            try:
                self.driver.close()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
        self._started = False

    def set_state(self, label: str | None) -> None:
        """Set the active label. ``None`` means "cut out" (no recording)."""
        with self._lock:
            if label is not None:
                self._state_changes.append((self._sample_count / self.sfreq, label))
            self._current_label = label

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    @property
    def current_label(self) -> str | None:
        with self._lock:
            return self._current_label

    def get_recording(self) -> tuple[np.ndarray, np.ndarray, list[tuple[float, str]]]:
        with self._lock:
            if self._signal_chunks:
                signal = np.concatenate(self._signal_chunks, axis=0)
            else:
                signal = np.zeros((0, self.montage.n_channels), dtype=np.float32)
            if self._time_chunks:
                time = np.concatenate(self._time_chunks, axis=0)
            else:
                time = np.zeros((0,), dtype=np.float64)
            changes = list(self._state_changes)
        # Make sure shapes are sane even if montage changed.
        if signal.ndim == 1:
            signal = signal.reshape(-1, self.montage.n_channels)
        if signal.shape[1] != self.montage.n_channels and signal.size == 0:
            signal = np.zeros((0, self.montage.n_channels), dtype=np.float32)
        return signal, time, changes

    # ----------------------------------------------------------------- thread
    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self.driver.read()
                except Exception:  # noqa: BLE001 - report via stop path
                    break
                if chunk is None:
                    chunk = np.empty((0, self.montage.n_channels), dtype=np.float32)
                chunk = np.asarray(chunk)
                if chunk.size:
                    n = chunk.shape[0]
                    with self._lock:
                        start = self._sample_count
                        times = (start + np.arange(n)) / self.sfreq
                        label = self._current_label
                        if label is not None:
                            self._signal_chunks.append(
                                np.array(chunk, dtype=np.float32, copy=True)
                            )
                            self._time_chunks.append(
                                np.array(times, dtype=np.float64, copy=True)
                            )
                        self._sample_count += n
                else:
                    self._sleep(self.poll_interval)
        finally:
            self._started = False
