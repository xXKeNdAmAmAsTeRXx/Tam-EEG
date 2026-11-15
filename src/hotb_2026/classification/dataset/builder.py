"""Turn continuous recordings into labeled, preprocessed training windows."""
from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import asdict, dataclass

import numpy as np

from hotb_2026.preprocessing import PreprocessingConfig, Preprocessor

from .schema import Recording

log = logging.getLogger(__name__)

_EPS = 1e-9
DEFAULT_LABEL_NAMES = ("low", "medium", "high")


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float
    state_change_s: float
    label: int
    block_index: int


def _change_times(recording: Recording) -> list[float]:
    return [change.time_s for change in recording.state_changes]


def _active_index(times: list[float], t: float) -> int | None:
    index = bisect_right(times, t) - 1
    return index if index >= 0 else None


def label_at(recording: Recording, t: float) -> int | None:
    """Label active at time ``t``, or ``None`` before the first state change."""
    times = _change_times(recording)
    index = _active_index(times, t)
    if index is None:
        return None
    return int(recording.state_changes[index].label)


def iter_windows(
    recording: Recording,
    window_s: float,
    stride_s: float | None = None,
    *,
    drop_boundary: bool = True,
    min_state_duration_s: float | None = None,
) -> tuple[list[Window], dict[str, int]]:
    """Plan windows and their labels without touching the signal."""
    stride_s = window_s if stride_s is None else stride_s
    if window_s <= 0 or stride_s <= 0:
        raise ValueError("window_s and stride_s must be positive")

    times = _change_times(recording)
    durations = [
        (times[index + 1] if index + 1 < len(times) else np.inf) - times[index]
        for index in range(len(times))
    ]

    signal = np.asarray(recording.signal)
    t_start = float(recording.time_s[0])
    t_end = t_start + signal.shape[1] / recording.sfreq

    windows: list[Window] = []
    dropped = {
        "unlabeled": 0,
        "boundary": 0,
        "short_state": 0,
        "too_short": 0,
    }

    start = t_start
    while start + window_s <= t_end + _EPS:
        end = start + window_s
        start_label = label_at(recording, start)
        end_label = label_at(recording, end - _EPS)

        if start_label is None or end_label is None:
            dropped["unlabeled"] += 1
            start += stride_s
            continue
        if drop_boundary and start_label != end_label:
            dropped["boundary"] += 1
            start += stride_s
            continue

        index = _active_index(times, start)
        assert index is not None  # label_at succeeded
        if (
            min_state_duration_s is not None
            and durations[index] < min_state_duration_s
        ):
            dropped["short_state"] += 1
            start += stride_s
            continue

        windows.append(
            Window(
                start_s=float(start),
                end_s=float(end),
                state_change_s=float(times[index]),
                label=int(start_label),
                block_index=int(index),
            )
        )
        start += stride_s

    return windows, dropped


def build_dataset(
    recordings: list[Recording] | tuple[Recording, ...],
    *,
    window_s: float = 10.0,
    stride_s: float | None = None,
    preprocess_config: PreprocessingConfig | None = None,
    drop_boundary: bool = True,
    min_state_duration_s: float | None = None,
    group_by: str = "auto",
    label_names: tuple[str, ...] | None = None,
) -> "EegDataset":
    """Build preprocessed windows from one or more recordings.

    Windows are cut on the raw time axis *before* preprocessing so every
    window is z-scored independently, matching live inference.

    ``group_by`` controls which independent unit cross-validation must keep
    intact: ``"session"`` (whole recording), ``"state_block"`` (maximal run of
    one label) or ``"auto"`` (session when there are at least five sessions,
    otherwise state block).
    """
    from .schema import EegDataset

    recordings = list(recordings)
    if not recordings:
        raise ValueError("recordings must not be empty")

    electrodes = tuple(recordings[0].electrodes)
    for recording in recordings[1:]:
        if tuple(recording.electrodes) != electrodes:
            raise ValueError(
                "all recordings must use the same electrodes in the same "
                f"order: {electrodes} vs {tuple(recording.electrodes)} "
                f"in session {recording.session_id!r}"
            )

    if group_by not in {"auto", "session", "state_block"}:
        raise ValueError(
            "group_by must be 'auto', 'session' or 'state_block', "
            f"got {group_by!r}"
        )

    if group_by == "auto":
        n_sessions = len({recording.session_id for recording in recordings})
        group_by = "session" if n_sessions >= 5 else "state_block"

    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    changes: list[float] = []

    totals = {
        "unlabeled": 0,
        "boundary": 0,
        "short_state": 0,
        "too_short": 0,
    }
    raw_sfreq = float(recordings[0].sfreq)
    for recording in recordings[1:]:
        if not np.isclose(recording.sfreq, raw_sfreq):
            raise ValueError(
                "all recordings must use the same raw sample rate: "
                f"{raw_sfreq:g} vs {recording.sfreq:g} in session "
                f"{recording.session_id!r}"
            )

    for recording in recordings:
        config = preprocess_config
        if config is None:
            config = PreprocessingConfig(sfreq=recording.sfreq)
        elif not np.isclose(config.sfreq, recording.sfreq):
            raise ValueError(
                f"preprocess_config.sfreq={config.sfreq} does not match "
                f"recording {recording.session_id!r} at {recording.sfreq} Hz"
            )
        preprocessor = Preprocessor(config)

        windows, dropped = iter_windows(
            recording,
            window_s,
            stride_s,
            drop_boundary=drop_boundary,
            min_state_duration_s=min_state_duration_s,
        )
        for key, value in dropped.items():
            totals[key] += value

        for window in windows:
            i0 = int(np.searchsorted(recording.time_s, window.start_s, side="left"))
            i1 = int(np.searchsorted(recording.time_s, window.end_s, side="left"))
            raw = np.asarray(recording.signal)[:, i0:i1]
            if raw.shape[1] < preprocessor.min_samples:
                totals["too_short"] += 1
                continue

            X.append(preprocessor.process(raw))
            y.append(window.label)
            starts.append(window.start_s)
            ends.append(window.end_s)
            changes.append(window.state_change_s)
            if group_by == "session":
                groups.append(recording.session_id)
            else:
                groups.append(f"{recording.session_id}:b{window.block_index}")

    if not X:
        raise ValueError(
            "no windows were produced; check window_s, labels and state changes"
        )

    if label_names is None:
        n_labels = max(y) + 1
        if n_labels == len(DEFAULT_LABEL_NAMES):
            label_names = DEFAULT_LABEL_NAMES
        else:
            label_names = tuple(f"class_{index}" for index in range(n_labels))

    preprocessing = asdict(
        preprocess_config
        if preprocess_config is not None
        else PreprocessingConfig(sfreq=float(raw_sfreq))
    )

    log.info(
        "Built %d windows from %d recordings (%d groups); dropped %s",
        len(X),
        len(recordings),
        len(set(groups)),
        totals,
    )

    return EegDataset(
        X=np.stack(X).astype(np.float32, copy=False),
        y=np.asarray(y, dtype=np.int8),
        groups=np.asarray(groups, dtype=str),
        window_start_s=np.asarray(starts, dtype=np.float64),
        window_end_s=np.asarray(ends, dtype=np.float64),
        state_change_s=np.asarray(changes, dtype=np.float64),
        electrodes=tuple(recordings[0].electrodes),
        sfreq=float(preprocessor.output_sfreq),
        raw_sfreq=float(raw_sfreq),
        window_s=float(window_s),
        label_names=tuple(label_names),
        preprocessing=preprocessing,
    )


def load_recordings(directory: str) -> list[Recording]:
    """Load every session directory under ``directory``."""
    from pathlib import Path

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"recordings directory not found: {root}")
    sessions = sorted(
        path for path in root.iterdir() if (path / "meta.json").exists()
    )
    if not sessions:
        raise FileNotFoundError(f"no sessions found under {root}")
    return [Recording.from_dir(path) for path in sessions]


__all__ = [
    "DEFAULT_LABEL_NAMES",
    "Window",
    "build_dataset",
    "iter_windows",
    "label_at",
    "load_recordings",
]
