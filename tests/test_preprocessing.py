from __future__ import annotations

import numpy as np
import pytest

from hotb_2026.preprocessing import (
    PreprocessingConfig,
    Preprocessor,
    WindowedPreprocessor,
)

SFREQ = 250.0


def _sine(
    freq: float, amplitude: float = 10.0, seconds: float = 10.0, sfreq: float = SFREQ
) -> np.ndarray:
    t = np.arange(round(seconds * sfreq)) / sfreq
    return amplitude * np.sin(2 * np.pi * freq * t)


def _amplitude(x: np.ndarray, freq: float, sfreq: float, channel: int = 0) -> float:
    series = np.asarray(x)[channel]
    n = series.shape[-1]
    spectrum = np.fft.rfft(series)
    k = int(round(freq * n / sfreq))
    return float(2 * np.abs(spectrum[k]) / n)


def _passthrough(sfreq: float = 10.0) -> Preprocessor:
    return Preprocessor(
        PreprocessingConfig(
            sfreq=sfreq,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=False,
            clip=None,
        )
    )


# ---- config ----


def test_config_rejects_invalid_bandpass() -> None:
    with pytest.raises(ValueError, match="bandpass"):
        PreprocessingConfig(bandpass=(45.0, 0.5))
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingConfig(bandpass=(0.5, 125.0))


def test_config_rejects_invalid_notch() -> None:
    with pytest.raises(ValueError, match="notch_freq"):
        PreprocessingConfig(notch_freq=125.0)
    with pytest.raises(ValueError, match="notch_q"):
        PreprocessingConfig(notch_q=0.0)


def test_config_rejects_invalid_target_rate() -> None:
    with pytest.raises(ValueError, match="target Nyquist"):
        PreprocessingConfig(bandpass=(0.5, 45.0), target_sfreq=80.0)
    with pytest.raises(ValueError, match="target_sfreq"):
        PreprocessingConfig(target_sfreq=0.0)


def test_config_rejects_invalid_clip_and_sfreq() -> None:
    with pytest.raises(ValueError, match="clip"):
        PreprocessingConfig(clip=0.0)
    with pytest.raises(ValueError, match="sfreq"):
        PreprocessingConfig(sfreq=0.0)


def test_config_validates_disabled_stage_parameters() -> None:
    # An invalid value is a mistake even if the stage is currently disabled.
    with pytest.raises(ValueError, match="notch_q"):
        PreprocessingConfig(notch_freq=None, notch_q=0.0)
    with pytest.raises(ValueError, match="filter_order"):
        PreprocessingConfig(bandpass=None, filter_order=0)


def test_config_bandpass_must_have_two_edges() -> None:
    with pytest.raises(ValueError, match="two edges"):
        PreprocessingConfig(bandpass=(1.0,))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sfreq", float("nan")),
        ("sfreq", float("inf")),
        ("notch_q", float("nan")),
        ("notch_freq", float("inf")),
        ("filter_order", float("nan")),
        ("target_sfreq", float("nan")),
        ("clip", float("inf")),
    ],
)
def test_config_rejects_non_finite_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        PreprocessingConfig(**{field: value})


def test_config_rejects_non_finite_bandpass_edges() -> None:
    with pytest.raises(ValueError, match="finite"):
        PreprocessingConfig(bandpass=(0.5, float("inf")))


def test_config_rejects_non_numeric_parameter() -> None:
    with pytest.raises(ValueError, match="finite number"):
        PreprocessingConfig(sfreq="250")  # type: ignore[arg-type]


def test_config_filter_order_must_be_whole_number() -> None:
    with pytest.raises(ValueError, match="whole number"):
        PreprocessingConfig(filter_order=4.5)  # type: ignore[arg-type]
    # ``butter`` accepts an integral float, so the config must too.
    PreprocessingConfig(filter_order=4.0)  # type: ignore[arg-type]


# ---- stages ----


def test_notch_attenuates_50_hz_and_keeps_alpha() -> None:
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=50.0,
            bandpass=None,
            target_sfreq=None,
            zscore=False,
            clip=None,
        )
    )
    raw = _sine(10.0) + _sine(50.0)
    cleaned = pre.process(raw[None, :])

    before_50 = _amplitude(raw[None, :], 50.0, SFREQ)
    after_50 = _amplitude(cleaned, 50.0, SFREQ)
    attenuation_db = 20 * np.log10(before_50 / after_50)

    assert attenuation_db > 20.0
    assert _amplitude(cleaned, 10.0, SFREQ) > 0.9 * _amplitude(raw[None, :], 10.0, SFREQ)


def test_bandpass_keeps_alpha_and_rejects_drift_and_high_frequencies() -> None:
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=(0.5, 45.0),
            target_sfreq=None,
            zscore=False,
            clip=None,
        )
    )
    raw = _sine(10.0, amplitude=10.0, seconds=20.0)
    raw += _sine(0.2, amplitude=10.0, seconds=20.0)
    raw += _sine(80.0, amplitude=10.0, seconds=20.0)
    cleaned = pre.process(raw[None, :])

    assert _amplitude(cleaned, 10.0, SFREQ) > 7.0
    assert _amplitude(cleaned, 0.2, SFREQ) < 0.1
    assert _amplitude(cleaned, 80.0, SFREQ) < 0.5


def test_resampling_changes_length_and_rate() -> None:
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=200.0,
            zscore=False,
            clip=None,
        )
    )
    raw = np.zeros((2, 1000), dtype=np.float64)
    cleaned = pre.process(raw)

    assert cleaned.shape == (2, 800)
    assert cleaned.dtype == np.float32
    assert pre.output_sfreq == 200.0


def test_zscore_standardizes_each_channel() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(3, 1000)) * np.array([[1.0], [5.0], [0.5]])
    raw = raw + np.array([[10.0], [-3.0], [100.0]])

    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=True,
            clip=None,
        )
    )
    cleaned = pre.process(raw)

    np.testing.assert_allclose(cleaned.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(cleaned.std(axis=1), 1.0, atol=1e-5)


def test_flat_channel_becomes_zeros_not_nan() -> None:
    raw = np.vstack([np.zeros(1000), np.full(1000, 5.0)])
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=True,
            clip=None,
        )
    )
    cleaned = pre.process(raw)

    assert np.all(cleaned == 0.0)
    assert np.all(np.isfinite(cleaned))


def test_clip_limits_outliers() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(2, 1000))
    raw[0, 123] = 1e6

    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=True,
            clip=15.0,
        )
    )
    cleaned = pre.process(raw)

    assert cleaned.max() == pytest.approx(15.0)
    assert cleaned.min() >= -15.0
    assert np.abs(cleaned).max() <= 15.0


def test_full_pipeline_preserves_channels_and_shape() -> None:
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(8, 1000))

    cleaned = Preprocessor().process(raw)

    assert cleaned.shape == (8, 800)
    assert cleaned.dtype == np.float32
    assert np.all(np.isfinite(cleaned))
    assert np.abs(cleaned).max() <= 15.0


def test_min_samples_is_exactly_the_filter_requirement() -> None:
    pre = Preprocessor()
    shortest = pre.min_samples

    pre.process(np.zeros((1, shortest), dtype=np.float64))
    with pytest.raises(ValueError, match="too short"):
        pre.process(np.zeros((1, shortest - 1), dtype=np.float64))


def test_min_samples_tracks_enabled_filters() -> None:
    notch_only = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=50.0,
            bandpass=None,
            target_sfreq=None,
            zscore=False,
            clip=None,
        )
    )
    no_filters = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=False,
            clip=None,
        )
    )
    assert no_filters.min_samples == 1
    assert 1 < notch_only.min_samples < Preprocessor().min_samples
    notch_only.process(np.zeros((1, notch_only.min_samples)))
    with pytest.raises(ValueError, match="too short"):
        notch_only.process(np.zeros((1, notch_only.min_samples - 1)))


def test_resampling_supports_upsampling() -> None:
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=200.0,
            notch_freq=None,
            bandpass=None,
            target_sfreq=250.0,
            zscore=False,
            clip=None,
        )
    )
    cleaned = pre.process(np.zeros((1, 1000)))

    assert cleaned.shape == (1, 1250)
    assert pre.output_sfreq == 250.0


def test_target_rate_equal_to_input_skips_resampling() -> None:
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=SFREQ,
            zscore=False,
            clip=None,
        )
    )
    cleaned = pre.process(np.zeros((1, 100)))

    assert cleaned.shape == (1, 100)
    assert pre.output_sfreq == SFREQ


def test_output_sfreq_without_resampling() -> None:
    pre = _passthrough(SFREQ)

    assert pre.output_sfreq == SFREQ


def test_output_sfreq_reports_input_rate_for_near_equal_target() -> None:
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=SFREQ + 1e-7,
            zscore=False,
            clip=None,
        )
    )
    cleaned = pre.process(np.zeros((1, 100)))

    # ``math.isclose`` skips resampling, so the actual rate stays at sfreq even
    # though the caller asked for a slightly different target.
    assert cleaned.shape == (1, 100)
    assert pre.output_sfreq == SFREQ


def test_clip_can_be_disabled() -> None:
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(1, 1000))
    raw[0, 0] = 1e6

    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=True,
            clip=None,
        )
    )
    cleaned = pre.process(raw)

    assert cleaned.max() > 15.0


def test_near_flat_channel_is_treated_as_flat() -> None:
    raw = np.zeros((1, 1000))
    raw[0, 500] = 1e-13

    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=SFREQ,
            notch_freq=None,
            bandpass=None,
            target_sfreq=None,
            zscore=True,
            clip=None,
        )
    )

    assert np.all(pre.process(raw) == 0.0)


def test_integer_input_is_converted_to_float32() -> None:
    pre = _passthrough(sfreq=10.0)
    cleaned = pre.process(np.zeros((2, 20), dtype=np.int16))

    assert cleaned.dtype == np.float32
    assert cleaned.shape == (2, 20)


def test_input_is_not_modified() -> None:
    raw = np.random.default_rng(4).normal(size=(2, 1000))
    original = raw.copy()

    Preprocessor().process(raw)

    np.testing.assert_array_equal(raw, original)


def test_invalid_windows_raise_clear_errors() -> None:
    pre = Preprocessor()
    with pytest.raises(ValueError, match="2-D"):
        pre.process(np.zeros(1000))
    with pytest.raises(ValueError, match="at least one channel"):
        pre.process(np.zeros((0, 1000)))
    with pytest.raises(ValueError, match="too short"):
        pre.process(np.zeros((2, 10)))
    with pytest.raises(ValueError, match="non-finite"):
        pre.process(np.full((2, 1000), np.nan))


def test_non_numeric_and_complex_windows_are_rejected() -> None:
    pre = _passthrough(sfreq=10.0)
    with pytest.raises(ValueError, match="numeric"):
        pre.process(np.array([["a"] * 20], dtype=object))
    with pytest.raises(ValueError, match="numeric"):
        pre.process(np.zeros((1, 20), dtype=complex))


# ---- windowing ----


def test_windowed_preprocessor_emits_one_result_per_window() -> None:
    results: list[tuple[np.ndarray, float]] = []
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda window, t: results.append((window.copy(), t)),
    )

    for i in range(1, 11):
        windowed(np.full((2, i), float(i), dtype=np.float32), i / 10)

    assert len(results) == 1
    assert results[0][0].shape == (2, 10)
    assert results[0][1] == pytest.approx(1.0)

    for i in range(11, 21):
        values = np.arange(i - 9, i + 1, dtype=np.float32)
        windowed(np.tile(values, (2, 1)), i / 10)

    assert len(results) == 2
    assert results[1][1] == pytest.approx(2.0)
    np.testing.assert_array_equal(results[1][0][0], np.arange(11, 21))


def test_windowed_preprocessor_resamples_the_output() -> None:
    results: list[np.ndarray] = []
    pre = Preprocessor(
        PreprocessingConfig(
            sfreq=10.0,
            notch_freq=None,
            bandpass=None,
            target_sfreq=5.0,
            zscore=False,
            clip=None,
        )
    )
    windowed = WindowedPreprocessor(
        pre, sfreq=10.0, window_s=1.0, on_window=lambda w, t: results.append(w)
    )
    windowed(np.zeros((2, 10), dtype=np.float32), 1.0)

    assert results[0].shape == (2, 5)


def test_windowed_preprocessor_counts_skipped_windows() -> None:
    results: list[np.ndarray] = []
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: results.append(w),
    )
    windowed(np.zeros((2, 30), dtype=np.float32), 3.0)

    assert len(results) == 1
    assert windowed.skipped_windows == 2


def test_windowed_preprocessor_rejects_short_streamer_window() -> None:
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: None,
    )
    with pytest.raises(RuntimeError, match="shorter"):
        windowed(np.zeros((2, 5), dtype=np.float32), 1.0)


def test_windowed_preprocessor_rejects_non_2d_window() -> None:
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: None,
    )
    with pytest.raises(RuntimeError, match="2-D"):
        windowed(np.zeros(10, dtype=np.float32), 1.0)


def test_windowed_preprocessor_rejects_non_finite_time() -> None:
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: None,
    )
    with pytest.raises(ValueError, match="finite stream time"):
        windowed(np.zeros((2, 10), dtype=np.float32), float("nan"))


def test_windowed_preprocessor_exposes_window_samples() -> None:
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=0.5,
        on_window=lambda w, t: None,
    )

    assert windowed.window_samples == 5


def test_windowed_preprocessor_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="sfreq"):
        WindowedPreprocessor(
            _passthrough(), sfreq=0.0, window_s=1.0, on_window=lambda w, t: None
        )
    with pytest.raises(ValueError, match="sfreq"):
        WindowedPreprocessor(
            _passthrough(),
            sfreq=float("nan"),
            window_s=1.0,
            on_window=lambda w, t: None,
        )
    with pytest.raises(ValueError, match="window_s"):
        WindowedPreprocessor(
            _passthrough(), sfreq=10.0, window_s=0.0, on_window=lambda w, t: None
        )
    with pytest.raises(ValueError, match="window_s"):
        WindowedPreprocessor(
            _passthrough(),
            sfreq=10.0,
            window_s=float("inf"),
            on_window=lambda w, t: None,
        )


def test_windowed_preprocessor_slices_the_exact_completed_window() -> None:
    results: list[np.ndarray] = []
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: results.append(w),
    )
    # A 2.2 s streamer window ending at sample 22. The first 1 s window is
    # skipped, the completed boundary at sample 20 is still in the buffer, so
    # the exact window [10, 20) is processed with no shift.
    values = np.arange(22, dtype=np.float32)
    windowed(np.tile(values, (2, 1)), 2.2)

    assert windowed.skipped_windows == 1
    assert windowed.shifted_windows == 0
    np.testing.assert_array_equal(results[0][0], np.arange(10, 20))


def test_windowed_preprocessor_falls_back_when_buffer_cannot_align() -> None:
    results: list[np.ndarray] = []
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: results.append(w),
    )
    # The streamer window is exactly 1 s and the chunk overshoots the boundary
    # by 2 samples, so the exact [0, 10) window is no longer available.
    values = np.arange(2, 12, dtype=np.float32)
    windowed(np.tile(values, (2, 1)), 1.2)

    assert windowed.shifted_windows == 1
    np.testing.assert_array_equal(results[0][0], values)


def test_windowed_preprocessor_resets_after_stream_restart() -> None:
    results: list[np.ndarray] = []
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: results.append(w),
    )
    windowed(np.full((2, 10), 1.0, dtype=np.float32), 1.0)
    windowed(np.full((2, 10), 2.0, dtype=np.float32), 2.0)
    assert len(results) == 2

    # Stream time moves backwards: this is a restart of the same streamer.
    windowed(np.full((2, 10), 3.0, dtype=np.float32), 0.1)
    assert len(results) == 2

    windowed(np.full((2, 10), 4.0, dtype=np.float32), 1.0)
    assert len(results) == 3
    np.testing.assert_array_equal(results[2], np.full((2, 10), 4.0))


def test_windowed_preprocessor_reset_clears_counters() -> None:
    results: list[np.ndarray] = []
    windowed = WindowedPreprocessor(
        _passthrough(),
        sfreq=10.0,
        window_s=1.0,
        on_window=lambda w, t: results.append(w),
    )
    windowed(np.zeros((2, 30), dtype=np.float32), 3.0)
    assert windowed.skipped_windows == 2

    windowed.reset()

    assert windowed.skipped_windows == 0
    assert windowed.shifted_windows == 0
