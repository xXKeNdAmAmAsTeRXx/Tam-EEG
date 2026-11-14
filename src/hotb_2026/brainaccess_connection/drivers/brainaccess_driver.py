"""BrainAccess BLE driver and SDK helpers."""
from __future__ import annotations

import collections
import logging
import threading

import numpy as np

from ..driver import DeviceError
from ..models.montage import Montage

log = logging.getLogger(__name__)

# Sample rates every BrainAccess model supports. The MINI supports only these.
HARDWARE_SFREQ = (250, 500)
_GAIN_BY_MULTIPLIER = (1, 2, 4, 6, 8, 12)


def _import_sdk():
    try:
        from brainaccess import core
        from brainaccess.core import eeg_channel
        from brainaccess.core.eeg_manager import EEGManager
        from brainaccess.core.gain_mode import GainMode
        from brainaccess.utils.exceptions import BrainAccessException
    except ImportError as exc:
        raise DeviceError(
            "BrainAccess SDK missing. Install it with: uv sync"
        ) from exc
    return core, eeg_channel, EEGManager, GainMode, BrainAccessException


def list_devices(adapter_index: int | None = None) -> list[str]:
    """Names of BrainAccess headsets found by a BLE scan (blocks ~5 seconds)."""
    core, *_ = _import_sdk()
    core.init()
    try:
        if adapter_index is not None:
            core.config_set_adapter_index(adapter_index)
        return [d.name for d in core.scan()]
    finally:
        try:
            core.close()
        except Exception:
            log.debug("core.close() failed after scan", exc_info=True)


class BrainAccessDriver:
    """Streams EEG from a BrainAccess headset over BLE using the SDK's core API.

    The SDK calls our chunk callback from its own thread. The callback only
    queues data; read() drains the queue, so this fits the pull-based Driver
    protocol.
    """

    def __init__(
        self,
        device_name: str,
        montage: Montage,
        *,
        adapter_index: int | None = None,   # Linux only: hci<N>
        chunk_size: int = 25,               # SDK default; allowed range 1-100
        gain: int | object | None = None,   # GainMode member or multiplier
        queue_limit: int = 4000,            # 4000 x 25 samples = 400 s at 250 Hz
    ):
        if not 1 <= chunk_size <= 100:
            raise ValueError(f"chunk_size must be in 1..100, got {chunk_size}")
        if isinstance(gain, int) and gain not in _GAIN_BY_MULTIPLIER:
            raise ValueError(
                f"gain must be one of {_GAIN_BY_MULTIPLIER} or a GainMode, got {gain}"
            )
        if round(montage.sfreq) not in HARDWARE_SFREQ:
            raise DeviceError(
                f"BrainAccess MINI supports 250 or 500 Hz, not {montage.sfreq:g} Hz. "
                "Use --simulate for other rates, or pass --sfreq 250 "
                "(or set MONTAGE.sfreq in config.py)."
            )
        self.device_name = device_name
        self.montage = montage
        self.sfreq = montage.sfreq
        self.adapter_index = adapter_index
        self.chunk_size = chunk_size
        self.gain = gain
        self._queue: collections.deque[np.ndarray] = collections.deque(maxlen=queue_limit)
        self._lock = threading.Lock()
        self._sdk = None
        self._ctx = None
        self._mgr = None
        self._src_rows: list[int] = []
        self._streaming = False
        self._disconnected = False
        self._error: str | None = None
        self._dropped_chunks = 0

    # ---- Driver protocol ----

    def open(self) -> None:
        self._sdk = _import_sdk()
        core, _, EEGManager, _, BrainAccessException = self._sdk
        try:
            core.init()
            if self.adapter_index is not None:
                core.config_set_adapter_index(self.adapter_index)
            core.config_set_chunk_size(self.chunk_size)
            self._ctx = EEGManager()
            self._mgr = self._ctx.__enter__()
            self._mgr.connect(self.device_name)
            self._mgr.set_callback_disconnect(self._on_disconnect)
            self._check_device()
        except DeviceError:
            self._teardown()
            raise
        except BrainAccessException as exc:
            self._teardown()
            raise DeviceError(
                f"Could not connect to {self.device_name!r}: {exc}"
            ) from exc
        except Exception as exc:
            self._teardown()
            raise DeviceError(
                f"Could not open {self.device_name!r}: {exc}"
            ) from exc

    def _check_device(self) -> None:
        assert self._mgr is not None
        n_electrodes = self._mgr.get_device_features().electrode_count()
        needed = max(e.pin for e in self.montage.electrodes) + 1
        if needed > n_electrodes:
            raise DeviceError(
                f"Montage uses electrode {needed - 1}, but the headset has "
                f"{n_electrodes}"
            )

    def start(self) -> None:
        if self._mgr is None or self._sdk is None:
            raise DeviceError("open() must be called before start()")
        _, ch, _, GainMode, BrainAccessException = self._sdk
        mgr = self._mgr
        gain = self._resolve_gain(GainMode)
        bias = self.montage.bias_electrode
        try:
            # Settings reset after every stream stop, so apply them before every
            # start.
            mgr.set_sample_rate(round(self.montage.sfreq))
            for e in self.montage.electrodes:
                channel = ch.ELECTRODE_MEASUREMENT + e.pin
                mgr.set_channel_enabled(channel, True)
                mgr.set_channel_gain(channel, gain)
            if bias is not None:
                mgr.set_channel_bias_feedback(
                    ch.ELECTRODE_MEASUREMENT + bias
                )
            mgr.set_callback_chunk(self._on_chunk)
            mgr.load_config()  # same order as the SDK examples
            self._src_rows = [
                mgr.get_channel_index(ch.ELECTRODE_MEASUREMENT + e.pin)
                for e in self.montage.electrodes
            ]
            with self._lock:
                self._queue.clear()
                self._disconnected = False
                self._error = None
                self._dropped_chunks = 0
                self._streaming = True
            mgr.start_stream()
        except BrainAccessException as exc:
            with self._lock:
                self._streaming = False
            raise DeviceError(f"Could not start the stream: {exc}") from exc
        except Exception as exc:
            with self._lock:
                self._streaming = False
            raise DeviceError(f"Could not start the stream: {exc}") from exc

    def _resolve_gain(self, GainMode):
        if self.gain is None:
            return GainMode.X8
        if isinstance(self.gain, int):
            return getattr(GainMode, f"X{self.gain}")
        return self.gain

    def read(self) -> np.ndarray:
        with self._lock:
            blocks = list(self._queue)
            self._queue.clear()
            dropped = self._dropped_chunks
            self._dropped_chunks = 0
            error = self._error
            streaming = self._streaming
            disconnected = self._disconnected
        if dropped:
            log.warning("Dropped %d chunk(s); the consumer is too slow", dropped)
        if blocks:
            return np.concatenate(blocks, axis=0)
        if error is not None:
            raise DeviceError(error)
        if streaming and disconnected:
            raise DeviceError("The headset disconnected")
        return np.empty((0, self.montage.n_channels), dtype=np.float32)

    def stop(self) -> None:
        if self._mgr is None:
            return
        try:
            # Delivers the final partial chunk before returning.
            self._mgr.stop_stream()
        except Exception as exc:  # noqa: BLE001 - stop must not raise
            log.warning("stop_stream failed: %s", exc)
        finally:
            with self._lock:
                self._streaming = False

    def close(self) -> None:
        self._teardown()

    # ---- internals ----

    def _on_chunk(self, chunk, chunk_size: int) -> None:
        # Runs on the SDK thread. Keep it short and make no device calls here.
        try:
            block = np.column_stack(
                [np.asarray(chunk[r], dtype=np.float32) for r in self._src_rows]
            )
        except (IndexError, TypeError, ValueError) as exc:
            with self._lock:
                self._error = f"Could not read a data chunk: {exc}"
            return
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped_chunks += 1
            self._queue.append(block)

    def _on_disconnect(self) -> None:
        with self._lock:
            self._disconnected = True

    def _teardown(self) -> None:
        if self._mgr is not None:
            try:
                self._mgr.stop_stream()
            except Exception:
                log.debug("stop_stream failed during teardown", exc_info=True)
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)  # destroys the manager
            except Exception:
                log.debug("manager teardown failed", exc_info=True)
        self._ctx = None
        self._mgr = None
        if self._sdk is not None:
            try:
                self._sdk[0].close()  # releases the BrainAccess library
            except Exception:
                log.debug("core.close() failed during teardown", exc_info=True)
        self._sdk = None
        with self._lock:
            self._streaming = False
            self._queue.clear()
