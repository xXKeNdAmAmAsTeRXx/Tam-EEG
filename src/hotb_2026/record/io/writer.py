"""Session output writing helpers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def write_session(
    out_dir: str | Path,
    session_idx: int,
    meta: dict[str, Any],
    signal: np.ndarray,
    time: np.ndarray,
    state_changes: list[tuple[float, str]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write one session folder and return its path.

    Refuses to overwrite an existing session folder unless ``overwrite=True``.
    """
    root = Path(out_dir)
    session_dir = root / f"session_{session_idx}"
    if session_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Session folder already exists: {session_dir}. "
            "Choose another session index or pass overwrite=True."
        )
    session_dir.mkdir(parents=True, exist_ok=overwrite)

    signal = np.asarray(signal)
    time = np.asarray(time)

    # Ensure 2D signal even when empty.
    if signal.size == 0:
        n_channels = int(meta.get("n_channels", 0) or len(meta.get("electrodes", [])))
        signal = np.zeros((0, n_channels), dtype=np.float32)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    np.save(session_dir / "signal.npy", signal)
    np.save(session_dir / "time.npy", time)

    meta_out = dict(meta)
    meta_out.setdefault("session_idx", session_idx)
    meta_out.setdefault("n_samples", int(signal.shape[0]))
    meta_out.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with (session_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2, ensure_ascii=False)

    with (session_dir / "state_changes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "label"])
        for t, label in state_changes:
            writer.writerow([repr(float(t)), label])

    return session_dir
