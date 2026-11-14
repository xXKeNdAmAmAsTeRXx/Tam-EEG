"""Fan-out a device driver into a rolling window and an LSL stream."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from ..drivers.driver import DeviceError, Driver
from ..models.montage import Montage

log = logging.getLogger(__name__)

Processor = Callable[[np.ndarray, float], None]


class EEGStreamer:
    """Polls a driver, keeps a rolling window, and publishes an LSL outlet.

    Parameters
    ----------
    driver:
        Device driver (``BrainAccessDriver`` or ``SimulatedDriver``).
    montage:
        Channel layout. The LSL stream has one channel per electrode.
    stream_name:
        Name advertised to LSL consumers.
    source_id:
        Value of the LSL ``source_id``; use a unique one per headset.
    window_s:
        Maximum length of the rolling window passed to processors.
    poll_interval:
        Sleep between empty driver reads.
    """

    def __init__(
        self,
        driver: Driver,
        montage: Montage,
        *,
        stream_name: str = "BrainAccess EEG",
        source_id: str = "brainaccess-eeg-1",
        window_s: float = 4.0,
        poll_interval: float = 0.005,
    ):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.driver = driver
        self.montage = montage
        self.stream_name = stream_name
        self.source_id = source_id
        self.window_s = window_s
        self.poll_interval = poll_interval

        self._window_samples = max(1, round(montage.sfreq * window_s))
        self._buffer = np.zeros(
            (montage.n_channels, self._window_samples), dtype=np.float32
        )
        self._write = 0
        self._filled = 0
        self._sample_count = 0

        self._lock = threading.Lock()
        self._processors: list[Processor] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._outlet = None
        self._running = False
        self.error: BaseException | None = None

    # ---- public API ----

    def add_processor(self, processor: Processor) -> None:
        """Register ``processor(window, t)``.

        ``window`` has shape ``(n_channels, n_samples)`` in montage order and is
        a private copy; ``t`` is the stream time in seconds. Before the window
        is full, ``n_samples`` is smaller than ``window_s * sfreq``.
        """
        with self._lock:
            self._processors.append(processor)

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Streamer is already running")
        try:
            from pylsl import StreamInfo, StreamOutlet, cf_float32
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DeviceError(
                "pylsl is missing. Install it with: uv sync"
            ) from exc

        info = StreamInfo(
            self.stream_name,
            "EEG",
            self.montage.n_channels,
            float(self.montage.sfreq),
            cf_float32,
            self.source_id,
        )
        desc = info.desc()
        channels = desc.append_child("channels")
        for name in self.montage.names:
            channel = channels.append_child("channel")
            channel.append_child_value("label", name)
            channel.append_child_value("unit", "microvolts")
            channel.append_child_value("type", "EEG")
        desc.append_child_value("manufacturer", "BrainAccess")
        desc.append_child_value("device", self.driver.device_name
                                if hasattr(self.driver, "device_name") else "simulated")

        self.error = None
        self._stop_event.clear()
        self._reset_buffer()
        self._outlet = StreamOutlet(info)

        try:
            self.driver.open()
            self.driver.start()
        except BaseException:
            self._outlet = None
            raise

        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="EEGStreamer", daemon=True
        )
        self._thread.start()
        log.info(
            "LSL stream %r started: %d channels at %g Hz",
            self.stream_name,
            self.montage.n_channels,
            self.montage.sfreq,
        )

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                log.warning("Streamer thread did not exit within 10 s")
            self._thread = None

        try:
            self.driver.stop()
            # stop() may deliver one final, shorter chunk.
            try:
                block = self.driver.read()
            except Exception:  # noqa: BLE001 - already stopping
                block = np.empty((0, self.montage.n_channels), dtype=np.float32)
            if block.size:
                try:
                    self._publish(block)
                except Exception:
                    log.exception("Could not publish the final chunk")
        finally:
            self.driver.close()
            self._outlet = None
            self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._filled >= self._window_samples

    @property
    def filled_samples(self) -> int:
        with self._lock:
            return self._filled

    def window(self) -> np.ndarray:
        """Copy of the current rolling window, oldest sample first."""
        with self._lock:
            if self._filled < self._window_samples:
                return self._buffer[:, : self._filled].copy()
            return np.concatenate(
                (self._buffer[:, self._write :], self._buffer[:, : self._write]),
                axis=1,
            )

    # ---- internals ----

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                block = self.driver.read()
                if block.size:
                    self._publish(block)
                else:
                    self._stop_event.wait(self.poll_interval)
        except Exception as exc:  # noqa: BLE001 - reported via .error
            self.error = exc
            log.error("Streamer stopped: %s", exc)
        finally:
            self._running = False

    def _publish(self, block: np.ndarray) -> None:
        self._outlet.push_chunk(block.astype(np.float32, copy=False))
        self._append(block)
        processors = []
        with self._lock:
            processors = list(self._processors)
            window = self._window_locked()
            t = self._sample_count / self.montage.sfreq
        for processor in processors:
            try:
                processor(window, t)
            except Exception:
                log.exception("Processor %r failed", processor)

    def _reset_buffer(self) -> None:
        with self._lock:
            self._buffer.fill(0)
            self._write = 0
            self._filled = 0
            self._sample_count = 0

    def _append(self, block: np.ndarray) -> None:
        x = np.ascontiguousarray(block.T, dtype=np.float32)
        n = x.shape[1]
        with self._lock:
            capacity = self._window_samples
            if n >= capacity:
                self._buffer[:] = x[:, -capacity:]
                self._write = 0
                self._filled = capacity
            else:
                end = self._write + n
                if end <= capacity:
                    self._buffer[:, self._write : end] = x
                else:
                    first = capacity - self._write
                    self._buffer[:, self._write :] = x[:, :first]
                    self._buffer[:, : n - first] = x[:, first:]
                self._write = end % capacity
                self._filled = min(capacity, self._filled + n)
            self._sample_count += n

    def _window_locked(self) -> np.ndarray:
        if self._filled < self._window_samples:
            return self._buffer[:, : self._filled].copy()
        return np.concatenate(
            (self._buffer[:, self._write :], self._buffer[:, : self._write]),
            axis=1,
        )
