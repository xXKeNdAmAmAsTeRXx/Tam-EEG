"""Shared driver interface.

Both ``BrainAccessDriver`` and ``SimulatedDriver`` implement the ``Driver``
protocol. This module stays free of device/SDK imports so the streamer and
tests can depend on it without the BrainAccess SDK installed.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class DeviceError(RuntimeError):
    """Raised when a driver cannot open, start, or read from a device."""


class Driver(Protocol):
    def open(self) -> None: ...

    def start(self) -> None: ...

    def read(self) -> np.ndarray:
        """New samples, shape (n_samples, n_channels), columns in montage order.

        Returns an empty array if nothing is ready yet.
        """
        ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
