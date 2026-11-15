from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hotb_2026.brainaccess_connection import MONTAGE
from hotb_2026.record import write_session


def test_write_session(tmp_path: Path):
    signal = np.zeros((5, MONTAGE.n_channels), dtype=np.float32)
    time_arr = np.arange(5, dtype=np.float64) / MONTAGE.sfreq
    changes = [(0.0, "rest")]
    meta = {
        "session_id": "s1",
        "subject_id": "subj",
        "electrodes": MONTAGE.names,
        "sfreq": MONTAGE.sfreq,
    }
    session_dir = write_session(tmp_path, 2, meta, signal, time_arr, changes)
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
