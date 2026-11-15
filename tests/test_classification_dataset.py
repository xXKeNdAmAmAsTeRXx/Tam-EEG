from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from hotb_2026.classification.dataset import (
    EegDataset,
    Recording,
    build_dataset,
    iter_windows,
    label_at,
    make_synthetic_recording,
)
from hotb_2026.preprocessing import PreprocessingConfig


@pytest.fixture(scope="module")
def recording() -> Recording:
    return make_synthetic_recording(
        duration_s=60.0,
        sfreq=250.0,
        state_duration_s=20.0,
        seed=1,
        session_id="session-a",
    )


@pytest.fixture(scope="module")
def dataset(recording: Recording) -> EegDataset:
    return build_dataset([recording], window_s=10.0)


def test_label_at_follows_state_changes(recording: Recording) -> None:
    assert label_at(recording, -1.0) is None
    assert label_at(recording, 0.0) == 0
    assert label_at(recording, 19.9) == 0
    assert label_at(recording, 20.0) == 1
    assert label_at(recording, 39.9) == 1
    assert label_at(recording, 40.0) == 2


def test_iter_windows_drops_boundary_windows(recording: Recording) -> None:
    windows, dropped = iter_windows(recording, window_s=15.0, stride_s=15.0)

    assert dropped["boundary"] == 2
    assert dropped["unlabeled"] == 0
    assert [window.start_s for window in windows] == [0.0, 45.0]
    assert [window.label for window in windows] == [0, 2]
    assert [window.state_change_s for window in windows] == [0.0, 40.0]


def test_build_dataset_windows_and_groups(dataset: EegDataset) -> None:
    assert dataset.n_windows == 6
    assert dataset.X.shape == (6, 8, 2000)
    assert dataset.sfreq == pytest.approx(200.0)
    assert dataset.raw_sfreq == pytest.approx(250.0)
    assert dataset.window_s == pytest.approx(10.0)
    assert dataset.label_names == ("low", "medium", "high")

    np.testing.assert_array_equal(dataset.y, [0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(
        dataset.window_end_s - dataset.window_start_s, np.full(6, 10.0)
    )

    # One session and fewer than five sessions -> state blocks.
    assert dataset.summary()["n_groups"] == 3
    assert all(group.startswith("session-a:b") for group in dataset.groups)
    np.testing.assert_allclose(dataset.state_change_s, [0, 0, 20, 20, 40, 40])


def test_build_dataset_can_group_by_session(
    recording: Recording,
) -> None:
    second = make_synthetic_recording(
        duration_s=60.0, state_duration_s=20.0, seed=2, session_id="session-b"
    )
    dataset = build_dataset(
        [recording, second], window_s=10.0, group_by="session"
    )
    assert set(dataset.groups) == {"session-a", "session-b"}
    assert dataset.n_windows == 12


def test_build_dataset_rejects_empty_result(recording: Recording) -> None:
    with pytest.raises(ValueError, match="no windows"):
        build_dataset([recording], window_s=100.0)


def test_build_dataset_rejects_mismatched_electrodes(
    recording: Recording,
) -> None:
    other = make_synthetic_recording(
        duration_s=60.0,
        state_duration_s=20.0,
        seed=3,
        session_id="session-c",
        electrodes=("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"),
    )
    reordered = make_synthetic_recording(
        duration_s=60.0,
        state_duration_s=20.0,
        seed=4,
        session_id="session-d",
        electrodes=tuple(reversed(other.electrodes)),
    )
    with pytest.raises(ValueError, match="same electrodes"):
        build_dataset([other, reordered], window_s=10.0)


def test_build_dataset_rejects_mismatched_sample_rates(
    recording: Recording,
) -> None:
    faster = make_synthetic_recording(
        duration_s=60.0,
        sfreq=200.0,
        state_duration_s=20.0,
        seed=5,
        session_id="session-e",
    )
    with pytest.raises(ValueError, match="same raw sample rate"):
        build_dataset([recording, faster], window_s=10.0)


def test_dataset_save_load_roundtrip(
    dataset: EegDataset, tmp_path
) -> None:
    path = tmp_path / "dataset"
    dataset.save(path)
    assert (tmp_path / "dataset.npz").exists()
    loaded = EegDataset.load(path)

    np.testing.assert_allclose(loaded.X, dataset.X)
    np.testing.assert_array_equal(loaded.y, dataset.y)
    np.testing.assert_array_equal(loaded.groups, dataset.groups)
    np.testing.assert_allclose(loaded.window_start_s, dataset.window_start_s)
    np.testing.assert_allclose(loaded.state_change_s, dataset.state_change_s)
    assert loaded.electrodes == dataset.electrodes
    assert loaded.label_names == dataset.label_names
    assert loaded.sfreq == pytest.approx(dataset.sfreq)
    assert loaded.preprocessing == json.loads(json.dumps(dataset.preprocessing))


def test_build_dataset_records_preprocessing(
    recording: Recording, tmp_path
) -> None:
    config = PreprocessingConfig(
        sfreq=recording.sfreq,
        notch_freq=60.0,
        notch_q=20.0,
        bandpass=(1.0, 40.0),
        filter_order=3,
        target_sfreq=200.0,
        zscore=False,
        clip=10.0,
    )
    dataset = build_dataset([recording], window_s=10.0, preprocess_config=config)

    expected = asdict(config)
    assert dataset.preprocessing == expected
    assert dataset.summary()["preprocessing"] == expected

    path = tmp_path / "preprocessed"
    dataset.save(path)
    # JSON has no tuples, so the band-pass edges come back as a list.
    assert EegDataset.load(path).preprocessing == json.loads(json.dumps(expected))


def test_load_dataset_without_preprocessing_is_backward_compatible(
    dataset: EegDataset, tmp_path
) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        X=dataset.X,
        y=dataset.y,
        groups=dataset.groups,
        window_start_s=dataset.window_start_s,
        window_end_s=dataset.window_end_s,
        state_change_s=dataset.state_change_s,
        electrodes=np.asarray(dataset.electrodes, dtype=str),
        label_names=np.asarray(dataset.label_names, dtype=str),
        sfreq=np.asarray(dataset.sfreq),
        raw_sfreq=np.asarray(dataset.raw_sfreq),
        window_s=np.asarray(dataset.window_s),
        format_version=np.asarray(1),
    )

    loaded = EegDataset.load(path)
    assert loaded.preprocessing == {}
    np.testing.assert_allclose(loaded.X, dataset.X)


def test_dataset_rejects_non_dict_preprocessing(dataset: EegDataset) -> None:
    with pytest.raises(ValueError, match="preprocessing must be a dict"):
        replace(dataset, preprocessing=["not", "a", "dict"])


def test_dataset_rejects_non_json_preprocessing(dataset: EegDataset) -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        replace(dataset, preprocessing={"bad": object()})


def test_recording_dir_roundtrip(recording: Recording, tmp_path) -> None:
    path = tmp_path / "session"
    recording.to_dir(path)
    loaded = Recording.from_dir(path)

    assert loaded.session_id == recording.session_id
    assert loaded.subject_id == recording.subject_id
    assert loaded.electrodes == recording.electrodes
    assert loaded.sfreq == pytest.approx(recording.sfreq)
    np.testing.assert_allclose(loaded.signal, recording.signal)
    np.testing.assert_allclose(loaded.time_s, recording.time_s)
    assert [change.time_s for change in loaded.state_changes] == [
        change.time_s for change in recording.state_changes
    ]
    assert [change.label for change in loaded.state_changes] == [
        change.label for change in recording.state_changes
    ]
