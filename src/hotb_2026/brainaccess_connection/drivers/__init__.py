"""Device drivers. The streamer only depends on the Driver protocol."""
from .driver import (
    HARDWARE_SFREQ,
    BrainAccessDriver,
    DeviceError,
    Driver,
    SimulatedDriver,
    list_devices,
)

__all__ = [
    "HARDWARE_SFREQ",
    "BrainAccessDriver",
    "DeviceError",
    "Driver",
    "SimulatedDriver",
    "list_devices",
]
