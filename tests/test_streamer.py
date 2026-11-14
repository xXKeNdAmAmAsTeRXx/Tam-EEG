from __future__ import annotations

import numpy as np

from hotb_2026.brainaccess_connection import EEGStreamer, Electrode, Montage


class FakeOutlet:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def push_chunk(self, chunk) -> None:
        self.chunks.append(np.asarray(chunk))


class FakeDriver:
    device_name = "fake"

    def open(self) -> None:
        pass

    def start(self) -> None:
        pass

    def read(self) -> np.ndarray:
        return np.empty((0, 2), dtype=np.float32)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _montage() -> Montage:
    return Montage(
        electrodes=(Electrode("A", 0), Electrode("B", 1)),
        sfreq=10.0,
        bias_electrode=None,
    )


def _streamer(window_s: float = 1.0) -> EEGStreamer:
    streamer = EEGStreamer(FakeDriver(), _montage(), window_s=window_s)
    streamer._outlet = FakeOutlet()
    return streamer


def test_window_is_empty_before_data() -> None:
    streamer = _streamer()
    assert streamer.window().shape == (2, 0)
    assert not streamer.is_full


def test_window_partial_then_full() -> None:
    streamer = _streamer()
    streamer._publish(np.zeros((3, 2), dtype=np.float32))
    assert streamer.filled_samples == 3
    assert streamer.window().shape == (2, 3)
    assert not streamer.is_full

    streamer._publish(np.ones((7, 2), dtype=np.float32))
    assert streamer.is_full
    assert streamer.window().shape == (2, 10)


def test_ring_buffer_keeps_last_samples_in_order() -> None:
    streamer = _streamer()
    for block_index in range(3):
        rows = np.arange(4 * block_index, 4 * block_index + 4, dtype=np.float32)
        block = np.repeat(rows[:, None], 2, axis=1)
        streamer._publish(block)

    window = streamer.window()
    expected = np.arange(2, 12, dtype=np.float32)
    np.testing.assert_allclose(window[0], expected)
    np.testing.assert_allclose(window[1], expected)


def test_block_larger_than_window_keeps_tail() -> None:
    streamer = _streamer()
    rows = np.arange(15, dtype=np.float32)
    streamer._publish(np.repeat(rows[:, None], 2, axis=1))

    np.testing.assert_allclose(streamer.window()[0], np.arange(5, 15))


def test_lsl_receives_samples_by_channels() -> None:
    streamer = _streamer()
    block = np.arange(8, dtype=np.float32).reshape(4, 2)
    streamer._publish(block)

    assert len(streamer._outlet.chunks) == 1
    np.testing.assert_allclose(streamer._outlet.chunks[0], block)
    assert streamer._outlet.chunks[0].shape == (4, 2)


def test_processors_get_window_and_stream_time() -> None:
    streamer = _streamer()
    calls = []

    def processor(window, t):
        calls.append((window.copy(), t))

    streamer.add_processor(processor)
    streamer._publish(np.ones((4, 2), dtype=np.float32))

    window, t = calls[0]
    assert window.shape == (2, 4)
    assert t == 0.4


def test_processor_failure_does_not_break_the_stream() -> None:
    streamer = _streamer()
    streamer.add_processor(lambda window, t: 1 / 0)
    streamer.add_processor(lambda window, t: None)

    streamer._publish(np.ones((4, 2), dtype=np.float32))

    assert streamer.filled_samples == 4
