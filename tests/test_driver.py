from __future__ import annotations

import logging

import numpy as np
import pytest

from hotb_2026.brainaccess_connection import (
    DEFAULT_MONTAGE,
    BrainAccessDriver,
    DeviceError,
    Electrode,
    Montage,
    SimulatedDriver,
)


def _driver(queue_limit: int = 10) -> BrainAccessDriver:
    driver = BrainAccessDriver("BA MINI 001", DEFAULT_MONTAGE, queue_limit=queue_limit)
    driver._src_rows = list(range(DEFAULT_MONTAGE.n_channels))
    return driver


def _chunk(values: list[list[float]]) -> list[np.ndarray]:
    return [np.asarray(column, dtype=np.float64) for column in values]


def test_on_chunk_maps_channels_in_montage_order() -> None:
    driver = _driver()
    block = [[float(ch), float(ch) + 0.5] for ch in range(8)]
    driver._on_chunk(_chunk(block), 2)

    data = driver.read()
    assert data.shape == (2, 8)
    assert data.dtype == np.float32
    np.testing.assert_allclose(data[0], np.arange(8, dtype=np.float32))
    np.testing.assert_allclose(data[1], np.arange(8, dtype=np.float32) + 0.5)


def test_read_empty_returns_correct_shape() -> None:
    data = _driver().read()
    assert data.shape == (0, 8)


def test_queue_overflow_is_counted_and_reported(caplog: pytest.LogCaptureFixture) -> None:
    driver = _driver(queue_limit=2)
    block = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    for _ in range(3):
        driver._on_chunk(_chunk(block), 1)

    with caplog.at_level(logging.WARNING, logger=driver.__class__.__module__):
        data = driver.read()
    assert data.shape == (2, 8)
    assert "Dropped 1 chunk" in caplog.text


def test_disconnect_is_reported() -> None:
    driver = _driver()
    driver._streaming = True
    driver._on_disconnect()
    with pytest.raises(DeviceError, match="disconnected"):
        driver.read()


def test_bad_chunk_layout_is_reported() -> None:
    driver = _driver()
    driver._src_rows = [99]
    driver._on_chunk([np.zeros(4)], 4)
    with pytest.raises(DeviceError, match="Could not read a data chunk"):
        driver.read()


def test_disconnect_after_stop_is_not_an_error() -> None:
    driver = _driver()
    driver._streaming = False
    driver._on_disconnect()
    assert driver.read().shape == (0, 8)


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        BrainAccessDriver("BA MINI 001", DEFAULT_MONTAGE, chunk_size=101)
    with pytest.raises(ValueError, match="gain"):
        BrainAccessDriver("BA MINI 001", DEFAULT_MONTAGE, gain=3)


def test_simulated_driver_shape_and_pacing() -> None:
    driver = SimulatedDriver(DEFAULT_MONTAGE, chunk_ms=4)
    with pytest.raises(DeviceError):
        driver.read()
    driver.start()
    try:
        data = driver.read()
    finally:
        driver.stop()
    assert data.shape == (1, 8)
    with pytest.raises(DeviceError):
        driver.read()


def test_simulated_driver_supports_any_rate() -> None:
    montage = Montage(
        electrodes=(Electrode("F3", 0),),
        sfreq=200.0,
        bias_electrode=None,
    )
    driver = SimulatedDriver(montage, chunk_ms=10)
    driver.start()
    try:
        data = driver.read()
    finally:
        driver.stop()
    assert data.shape == (2, 1)
