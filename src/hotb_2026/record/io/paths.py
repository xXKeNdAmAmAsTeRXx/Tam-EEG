"""Session folder layout helpers and collision checks."""

from __future__ import annotations

import re
from pathlib import Path

_SESSION_DIR = re.compile(r"^session_(\d+)$")


def subject_dir(out_dir: str | Path, subject_id: str) -> Path:
    """Return the per-subject folder under the output root."""
    return Path(out_dir) / subject_id


def session_dir(subject_root: str | Path, session_idx: int) -> Path:
    """Return the folder for one session of a subject."""
    return Path(subject_root) / f"session_{session_idx}"


def existing_session_indices(subject_root: str | Path) -> list[int]:
    """Return the session indices already present for a subject, sorted."""
    root = Path(subject_root)
    if not root.is_dir():
        return []
    indices: list[int] = []
    for child in root.iterdir():
        match = _SESSION_DIR.match(child.name)
        if match is not None and child.is_dir():
            indices.append(int(match.group(1)))
    return sorted(indices)


def next_free_session_idx(subject_root: str | Path) -> int:
    """Return the first index after the highest existing session."""
    return max(existing_session_indices(subject_root), default=0) + 1


def ensure_session_available(subject_root: str | Path, session_idx: int) -> None:
    """Raise if ``session_<session_idx>`` already exists for the subject."""
    target = session_dir(subject_root, session_idx)
    if target.exists():
        raise FileExistsError(
            f"Session {session_idx} already exists for this subject ({target}). "
            f"Choose an unused session index (next free: {next_free_session_idx(subject_root)})."
        )


def ensure_sessions_available(
    subject_root: str | Path, start_idx: int, count: int
) -> None:
    """Raise if any of ``start_idx .. start_idx + count - 1`` already exists."""
    if count < 1:
        raise ValueError("count must be >= 1")
    occupied = [
        idx
        for idx in range(start_idx, start_idx + count)
        if session_dir(subject_root, idx).exists()
    ]
    if occupied:
        listed = ", ".join(str(idx) for idx in occupied)
        raise FileExistsError(
            f"Session(s) {listed} already exist for this subject ({Path(subject_root)}). "
            f"Choose an unused session index (next free: {next_free_session_idx(subject_root)})."
        )
