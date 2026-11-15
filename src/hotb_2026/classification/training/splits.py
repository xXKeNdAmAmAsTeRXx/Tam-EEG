"""Grouped train/test splits that keep leakage out of training.

Windows from the same recording (or from the same state block) overlap in
time. If two adjacent windows end up in different folds the model can memorize
near-duplicates, so every split keeps whole groups together.

* :func:`group_holdout` picks groups for the outer, honest held-out set.
* :func:`inner_cv` builds the grouped cross-validation used by Optuna.

Both functions are deterministic for a fixed ``seed``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupHoldout:
    """Indices of the outer train/test split plus the groups held out."""

    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[str, ...]

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)


def _validate(groups: np.ndarray, y: np.ndarray) -> None:
    if groups.shape != y.shape:
        raise ValueError(
            f"groups and y must have the same shape, got {groups.shape} "
            f"and {y.shape}"
        )
    if groups.ndim != 1:
        raise ValueError(f"groups must be 1-D, got shape {groups.shape}")
    if groups.size == 0:
        raise ValueError("groups and y must not be empty")


def _class_coverage(
    y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> tuple[int, int]:
    """Return ``(covered, total)`` classes present in both parts."""
    classes = np.unique(y)
    covered = sum(
        1
        for label in classes
        if np.any(y[train_mask] == label) and np.any(y[test_mask] == label)
    )
    return covered, int(classes.size)


def group_holdout(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 0,
    max_attempts: int = 64,
) -> GroupHoldout:
    """Hold out whole groups, trying to preserve every class on both sides.

    Stratified group splitting is not exposed by scikit-learn for a single
    train/test split, so this samples candidate group subsets and keeps the
    one with the best class coverage and the closest test fraction.
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    _validate(groups, y)

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    unique_groups, group_index = np.unique(groups, return_inverse=True)
    n_groups = int(unique_groups.size)
    if n_groups < 2:
        raise ValueError(
            "need at least two groups for a held-out split, got "
            f"{n_groups}; add recordings or use group_by='state_block'"
        )
    if np.unique(y).size < 2:
        raise ValueError("need at least two classes to train a classifier")

    n_test_groups = int(round(test_size * n_groups))
    n_test_groups = max(1, min(n_groups - 1, n_test_groups))

    rng = np.random.default_rng(seed)
    best: tuple[tuple[int, float], tuple[str, ...]] | None = None

    # The first candidate is deterministic (evenly spaced groups); the rest are
    # random permutations. Keeping the best of several candidates is cheap
    # compared with training and avoids a pathological single-group holdout.
    candidates: list[np.ndarray] = []
    stride = max(1, n_groups // n_test_groups)
    candidates.append(np.arange(n_groups)[::stride][:n_test_groups])
    for _ in range(max_attempts - 1):
        candidates.append(rng.permutation(n_groups)[:n_test_groups])

    for selection in candidates:
        test_groups = unique_groups[np.sort(selection)]
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue

        covered, total = _class_coverage(y, train_mask, test_mask)
        fraction = float(test_mask.mean())
        # Prefer full class coverage first, then a fraction near test_size.
        score = (total - covered, abs(fraction - test_size))
        if best is None or score < best[0]:
            best = (score, tuple(str(group) for group in test_groups))

    if best is None:  # pragma: no cover - candidates always contain a valid split
        raise ValueError("could not build a grouped holdout split")

    test_groups = best[1]
    test_mask = np.isin(groups, test_groups)
    train_mask = ~test_mask
    covered, total = _class_coverage(y, train_mask, test_mask)
    if covered < total:
        log.warning(
            "Held-out split covers %d/%d classes; metrics that need every "
            "class (e.g. macro averages) will treat missing classes as zero",
            covered,
            total,
        )

    return GroupHoldout(
        train=np.flatnonzero(train_mask),
        test=np.flatnonzero(test_mask),
        test_groups=test_groups,
    )


def inner_cv(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[StratifiedGroupKFold, int]:
    """Return a working :class:`StratifiedGroupKFold` and its effective folds.

    ``n_splits`` is reduced automatically when there are fewer groups than
    requested (for example three state blocks).
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    _validate(groups, y)

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if np.unique(y).size < 2:
        raise ValueError("need at least two classes for cross-validation")

    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        raise ValueError(
            "need at least two groups for cross-validation, got "
            f"{n_groups}; add recordings or use group_by='state_block'"
        )

    last_error: Exception | None = None
    for folds in range(min(n_splits, n_groups), 1, -1):
        cv = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        try:
            splits = list(cv.split(np.zeros_like(y), y, groups=groups))
        except ValueError as exc:  # too few members for a stratification
            last_error = exc
            continue
        if all(len(train) and len(test) for train, test in splits):
            if folds != n_splits:
                log.info(
                    "Using %d folds instead of %d (only %d groups available)",
                    folds,
                    n_splits,
                    n_groups,
                )
            return cv, folds

    raise ValueError(
        "could not build a grouped cross-validation split with the available "
        f"groups: {last_error}"
    )


__all__ = ["GroupHoldout", "group_holdout", "inner_cv"]
