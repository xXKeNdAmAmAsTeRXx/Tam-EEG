from __future__ import annotations

import pytest

from hotb_2026.brainaccess_connection import (
    DEVICE_NAME,
    HARDWARE_SFREQ,
    MONTAGE,
    BrainAccessDriver,
    DeviceError,
    Electrode,
    Montage,
)


def test_package_exposes_config() -> None:
    assert DEVICE_NAME
    assert isinstance(MONTAGE, Montage)
    assert MONTAGE.n_channels == 8
    assert [e.pin for e in MONTAGE.electrodes] == list(range(8))
    assert MONTAGE.sfreq > 0


def test_config_bias_is_valid() -> None:
    assert MONTAGE.bias_electrode is None or (
        0 <= MONTAGE.bias_electrode < MONTAGE.n_channels
    )


def test_hardware_rejects_unsupported_sample_rate() -> None:
    montage = Montage(
        electrodes=(Electrode("F3", 0),),
        sfreq=200.0,
        bias_electrode=None,
    )
    assert round(montage.sfreq) not in HARDWARE_SFREQ
    with pytest.raises(DeviceError, match="250 or 500 Hz"):
        BrainAccessDriver("BA MINI 001", montage)
