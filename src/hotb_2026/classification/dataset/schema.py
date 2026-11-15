"""Data model shared by the training pipeline.

A recording is a continuous signal with one timestamp per sample
(``time_s`` — the *current time* of the format) and a sorted list of state
changes (``time_s`` — *when the label changes*). The recording's channel order
is described by electrode names.

``EegDataset`` is the preprocessed, windowed form used for training. It keeps
the raw time fields so every window can be traced back to the stream timeline.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
_EPS = 1e-9


def _npz_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def _require_finite(name: str, value: float) -> None:
    try:
        finite = math.isfinite(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not finite:
        raise ValueError(f"{name} must be finite, got {value}")


def _normalize_signal(
    signal: np.ndarray, n_channels: int, layout: str
) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(
            f"signal must be 2-D, got shape {signal.shape}"
        )
    if layout == "samples_by_channels":
        if signal.shape[1] != n_channels:
            raise ValueError(
                f"signal has {signal.shape[1]} columns but "
                f"{n_channels} electrodes were declared"
            )
        signal = signal.T
    elif layout == "channels_by_samples":
        if signal.shape[0] != n_channels:
            raise ValueError(
                f"signal has {signal.shape[0]} rows but "
                f"{n_channels} electrodes were declared"
            )
    else:
        raise ValueError(
            "signal_layout must be 'samples_by_channels' or "
            f"'channels_by_samples', got {layout!r}"
        )
    return np.ascontiguousarray(signal)


@dataclass(frozen=True)
class StateChange:
    """The label becomes ``label`` at stream time ``time_s``."""

    time_s: float
    label: int

    def __post_init__(self) -> None:
        _require_finite("state change time", self.time_s)
        if isinstance(self.label, bool) or not isinstance(
            self.label, (int, np.integer)
        ):
            raise ValueError(f"label must be an integer, got {self.label!r}")


@dataclass(frozen=True)
class Recording:
    """Continuous EEG with per-sample time and label change points.

    ``signal`` is stored as ``(n_channels, n_samples)`` in electrode order.
    """

    session_id: str
    subject_id: str
    electrodes: tuple[str, ...]
    signal: np.ndarray
    time_s: np.ndarray
    state_changes: tuple[StateChange, ...]
    sfreq: float

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.electrodes:
            raise ValueError("electrodes must not be empty")
        if len(set(self.electrodes)) != len(self.electrodes):
            raise ValueError("electrodes must be unique")

        signal = np.asarray(self.signal)
        if signal.ndim != 2:
            raise ValueError(f"signal must be 2-D, got shape {signal.shape}")
        if signal.shape[0] != len(self.electrodes):
            raise ValueError(
                f"signal has {signal.shape[0]} channels but "
                f"{len(self.electrodes)} electrodes were declared"
            )
        if not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
            raise ValueError(
                f"signal must contain real numeric values, got {signal.dtype}"
            )
        if not np.all(np.isfinite(signal)):
            raise ValueError("signal contains non-finite values (NaN or Inf)")

        time_s = np.asarray(self.time_s, dtype=np.float64)
        if time_s.ndim != 1:
            raise ValueError(f"time_s must be 1-D, got shape {time_s.shape}")
        if time_s.shape[0] != signal.shape[1]:
            raise ValueError(
                f"time_s has {time_s.shape[0]} entries but signal has "
                f"{signal.shape[1]} samples"
            )
        if not np.all(np.isfinite(time_s)):
            raise ValueError("time_s contains non-finite values")
        if time_s.shape[0] > 1 and not np.all(np.diff(time_s) > 0):
            raise ValueError("time_s must be strictly increasing")

        _require_finite("sfreq", self.sfreq)
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")
        if time_s.shape[0] > 1:
            estimated = 1.0 / float(np.median(np.diff(time_s)))
            if not math.isclose(estimated, self.sfreq, rel_tol=0.01):
                raise ValueError(
                    f"sfreq={self.sfreq:g} does not match the timestamps "
                    f"(estimated {estimated:.3f} Hz)"
                )

        if not self.state_changes:
            raise ValueError("state_changes must not be empty")
        previous = -math.inf
        for change in self.state_changes:
            if change.time_s < previous:
                raise ValueError("state_changes must be sorted by time")
            previous = change.time_s

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dir(cls, path: str | Path) -> "Recording":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        electrodes = tuple(str(name) for name in meta["electrodes"])

        layout = meta.get("signal_layout", "samples_by_channels")
        signal = _normalize_signal(np.load(path / "signal.npy"), len(electrodes), layout)

        sfreq = float(meta["sfreq"])
        time_path = path / "time.npy"
        if time_path.exists():
            time_s = np.load(time_path)
        else:
            start = float(meta.get("start_time_s", 0.0))
            time_s = start + np.arange(signal.shape[1]) / sfreq

        changes = []
        with (path / "state_changes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                changes.append(
                    StateChange(
                        time_s=float(row["time"]),
                        label=int(row["label"]),
                    )
                )
        changes.sort(key=lambda change: change.time_s)

        return cls(
            session_id=str(meta["session_id"]),
            subject_id=str(meta["subject_id"]),
            electrodes=electrodes,
            signal=signal,
            time_s=np.asarray(time_s, dtype=np.float64),
            state_changes=tuple(changes),
            sfreq=sfreq,
        )

    def to_dir(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": FORMAT_VERSION,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "electrodes": list(self.electrodes),
            "sfreq": self.sfreq,
            "signal_layout": "samples_by_channels",
            "time_units": "s",
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        np.save(path / "signal.npy", np.asarray(self.signal).T)
        np.save(path / "time.npy", np.asarray(self.time_s, dtype=np.float64))
        with (path / "state_changes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "label"])
            for change in self.state_changes:
                writer.writerow([change.time_s, change.label])


@dataclass
class EegDataset:
    """Preprocessed windows ready for embedding and classification.

    ``X`` is ``(n_windows, n_channels, n_samples)`` at ``sfreq`` (the
    preprocessing output rate). ``window_end_s`` is the *current time* of the
    window and ``state_change_s`` is when its label became active.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    window_start_s: np.ndarray
    window_end_s: np.ndarray
    state_change_s: np.ndarray
    electrodes: tuple[str, ...]
    sfreq: float
    raw_sfreq: float
    window_s: float
    label_names: tuple[str, ...]
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3-D (n, n_channels, n_samples), got {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("dataset contains no windows")
        if X.shape[1] != len(self.electrodes):
            raise ValueError(
                f"X has {X.shape[1]} channels but "
                f"{len(self.electrodes)} electrodes were declared"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")

        n = X.shape[0]
        for name, array in (
            ("y", self.y),
            ("groups", self.groups),
            ("window_start_s", self.window_start_s),
            ("window_end_s", self.window_end_s),
            ("state_change_s", self.state_change_s),
        ):
            if np.asarray(array).shape != (n,):
                raise ValueError(
                    f"{name} must have shape {(n,)}, got {np.asarray(array).shape}"
                )

        labels = np.asarray(self.y)
        if labels.dtype.kind not in "iu":
            if not np.all(np.equal(np.mod(labels, 1), 0)):
                raise ValueError("y must contain integer labels")
        for value in np.unique(labels):
            if int(value) < 0 or int(value) >= len(self.label_names):
                raise ValueError(
                    f"label {value} is outside label_names "
                    f"(0..{len(self.label_names) - 1})"
                )
        if self.sfreq <= 0 or self.raw_sfreq <= 0 or self.window_s <= 0:
            raise ValueError("sfreq, raw_sfreq and window_s must be positive")

    @property
    def n_windows(self) -> int:
        return int(self.X.shape[0])

    def summary(self) -> dict:
        labels = np.asarray(self.y).astype(int)
        groups = np.asarray(self.groups)
        return {
            "n_windows": self.n_windows,
            "n_channels": int(self.X.shape[1]),
            "n_samples": int(self.X.shape[2]),
            "sfreq": self.sfreq,
            "raw_sfreq": self.raw_sfreq,
            "window_s": self.window_s,
            "class_counts": {
                self.label_names[index]: int(count)
                for index, count in enumerate(
                    np.bincount(labels, minlength=len(self.label_names))
                )
            },
            "n_groups": int(np.unique(groups).size),
            "groups": sorted(str(group) for group in np.unique(groups)),
            "label_names": list(self.label_names),
            "electrodes": list(self.electrodes),
        }

    def save(self, path: str | Path) -> None:
        path = _npz_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=np.asarray(self.X, dtype=np.float32),
            y=np.asarray(self.y).astype(np.int8),
            groups=np.asarray(self.groups).astype(str),
            window_start_s=np.asarray(self.window_start_s, dtype=np.float64),
            window_end_s=np.asarray(self.window_end_s, dtype=np.float64),
            state_change_s=np.asarray(self.state_change_s, dtype=np.float64),
            electrodes=np.asarray(self.electrodes, dtype=str),
            label_names=np.asarray(self.label_names, dtype=str),
            sfreq=np.asarray(self.sfreq),
            raw_sfreq=np.asarray(self.raw_sfreq),
            window_s=np.asarray(self.window_s),
            format_version=np.asarray(self.format_version),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EegDataset":
        with np.load(_npz_path(path), allow_pickle=False) as data:
            return cls(
                X=data["X"],
                y=data["y"],
                groups=data["groups"],
                window_start_s=data["window_start_s"],
                window_end_s=data["window_end_s"],
                state_change_s=data["state_change_s"],
                electrodes=tuple(str(name) for name in data["electrodes"]),
                label_names=tuple(str(name) for name in data["label_names"]),
                sfreq=float(data["sfreq"]),
                raw_sfreq=float(data["raw_sfreq"]),
                window_s=float(data["window_s"]),
                format_version=int(data["format_version"]),
            )


__all__ = [
    "FORMAT_VERSION",
    "EegDataset",
    "Recording",
    "StateChange",
]
