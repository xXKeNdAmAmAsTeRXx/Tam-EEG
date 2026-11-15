"""Parameters for one training run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for :func:`hotb_2026.classification.training.train`.

    ``window_s`` and ``stride_s`` describe how raw recordings are turned into
    windows. They are ignored when a pre-built ``EegDataset`` is supplied.
    """

    window_s: float = 10.0
    stride_s: float | None = None
    n_splits: int = 5
    n_trials: int = 60
    seed: int = 0
    test_size: float = 0.2
    metric: str = "balanced_accuracy"
    group_by: str = "auto"
    drop_boundary: bool = True
    probability: bool = True
    timeout_s: float | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.stride_s is not None and self.stride_s <= 0:
            raise ValueError("stride_s must be positive")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if self.group_by not in {"auto", "session", "state_block"}:
            raise ValueError(
                "group_by must be 'auto', 'session' or 'state_block', "
                f"got {self.group_by!r}"
            )
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


__all__ = ["TrainingConfig"]
