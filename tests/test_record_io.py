from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hotb_2026.brainaccess_connection import MONTAGE
from hotb_2026.record import write_session
from hotb_2026.record.io.paths import (
    ensure_session_available,
    ensure_sessions_available,
    existing_session_indices,
    next_free_session_idx,
    subject_dir,
)


def _meta() -> dict:
    return {
        "session_id": "s1",
        "subject_id": "subj",
        "electrodes": MONTAGE.names,
        "sfreq": MONTAGE.sfreq,
        "n_channels": MONTAGE.n_channels,
    }


def test_write_session(tmp_path: Path):
    signal = np.zeros((5, MONTAGE.n_channels), dtype=np.float32)
    time_arr = np.arange(5, dtype=np.float64) / MONTAGE.sfreq
    changes = [(0.0, "rest")]
    session_dir = write_session(tmp_path, 2, _meta(), signal, time_arr, changes)
    assert session_dir.name == "session_2"
    assert (session_dir / "signal.npy").exists()
    assert (session_dir / "time.npy").exists()
    assert (session_dir / "meta.json").exists()
    assert (session_dir / "state_changes.csv").exists()

    loaded_meta = json.loads((session_dir / "meta.json").read_text())
    assert loaded_meta["session_id"] == "s1"
    assert loaded_meta["electrodes"] == MONTAGE.names
    assert np.load(session_dir / "signal.npy").shape == (5, MONTAGE.n_channels)
    csv_text = (session_dir / "state_changes.csv").read_text()
    assert "time,label" in csv_text
    assert "rest" in csv_text


def test_write_session_refuses_to_overwrite(tmp_path: Path):
    signal = np.zeros((5, MONTAGE.n_channels), dtype=np.float32)
    time_arr = np.arange(5, dtype=np.float64) / MONTAGE.sfreq
    write_session(tmp_path, 1, _meta(), signal, time_arr, [(0.0, "rest")])

    with pytest.raises(FileExistsError):
        write_session(tmp_path, 1, _meta(), signal, time_arr, [(0.0, "1back")])

    replaced = np.ones((7, MONTAGE.n_channels), dtype=np.float32)
    session_dir = write_session(
        tmp_path, 1, _meta(), replaced, time_arr[:7], [], overwrite=True
    )
    assert np.load(session_dir / "signal.npy").shape == (7, MONTAGE.n_channels)


def test_subject_scoped_session_paths(tmp_path: Path):
    subject_root = subject_dir(tmp_path, "subj")
    assert subject_root == tmp_path / "subj"
    assert next_free_session_idx(subject_root) == 1

    (subject_root / "session_1").mkdir(parents=True)
    (subject_root / "session_3").mkdir()
    (subject_root / "notes").mkdir()

    assert existing_session_indices(subject_root) == [1, 3]
    assert next_free_session_idx(subject_root) == 4

    with pytest.raises(FileExistsError, match="next free: 4"):
        ensure_session_available(subject_root, 3)

    ensure_session_available(subject_root, 2)
    ensure_sessions_available(subject_root, 2, 1)
    with pytest.raises(FileExistsError, match="1, 3"):
        ensure_sessions_available(subject_root, 1, 3)
