from __future__ import annotations

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
