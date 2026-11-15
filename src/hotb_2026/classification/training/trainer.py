"""End-to-end training: embeddings -> grouped holdout -> Optuna -> final SVM."""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from hotb_2026.features import ReveEncoder, load_or_encode
from hotb_2026.reporting.metrics import (
    align_probabilities,
    classification_metrics,
)

from .config import TrainingConfig
from .search import make_estimator, normalize_params, run_study
from .splits import group_holdout, inner_cv

log = logging.getLogger(__name__)


def embed_dataset(
    dataset: Any,
    encoder: Any | None = None,
    *,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode every window in ``dataset`` and return embeddings plus metadata.

    With ``cache_dir`` the embeddings are cached on disk and reused across
    runs (and Optuna trials never recompute them anyway).
    """
    if not hasattr(dataset, "X") or not hasattr(dataset, "electrodes"):
        raise TypeError("dataset must be an EegDataset-like object")
    if encoder is None:
        encoder = ReveEncoder(dataset.electrodes)
    if not hasattr(encoder, "encode"):
        raise TypeError("encoder must expose an encode(X) method")
    if tuple(getattr(encoder, "electrodes", ())) != tuple(dataset.electrodes):
        raise ValueError(
            "encoder electrodes do not match the dataset channel order: "
            f"{getattr(encoder, 'electrodes', None)} vs "
            f"{dataset.electrodes}"
        )

    model_name = getattr(encoder, "model_name", "unknown")
    meta: dict[str, Any] = {
        "model_name": model_name,
        "electrodes": list(dataset.electrodes),
        "n_windows": dataset.n_windows,
        "window_s": float(dataset.window_s),
        "sfreq": float(dataset.sfreq),
        "raw_sfreq": float(dataset.raw_sfreq),
    }

    if cache_dir is not None:
        embeddings = load_or_encode(
            cache_dir,
            encoder,
            dataset.X,
            model_name=model_name,
            extra={
                "window_s": float(dataset.window_s),
                "sfreq": float(dataset.sfreq),
                "format_version": int(dataset.format_version),
            },
        )
        meta["cache_dir"] = str(cache_dir)
    else:
        log.info("Encoding %d windows", dataset.n_windows)
        embeddings = encoder.encode(dataset.X)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != dataset.n_windows:
        raise ValueError(
            "encoder returned embeddings with shape "
            f"{embeddings.shape}; expected ({dataset.n_windows}, dim)"
        )
    meta["embedding_dim"] = int(embeddings.shape[1])
    meta["shape"] = list(embeddings.shape)
    return embeddings, meta


def _calibration_folds(y: np.ndarray, maximum: int = 5) -> int:
    counts = np.bincount(np.asarray(y).astype(int))
    counts = counts[counts > 0]
    if counts.size == 0:
        return 2
    return int(max(2, min(maximum, counts.min())))


def _fit(
    params: dict[str, Any],
    embeddings: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> Any:
    estimator = make_estimator(
        params,
        seed=config.seed,
        probability=config.probability,
        calibration_cv=_calibration_folds(y),
    )
    estimator.fit(embeddings, y)
    return estimator


def train_from_embeddings(
    embeddings: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    config: TrainingConfig | None = None,
    label_names: tuple[str, ...] | list[str] = ("low", "medium", "high"),
    dataset_summary: dict[str, Any] | None = None,
    embedding_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate on precomputed embeddings.

    Returns a plain ``dict``. The fitted pipelines are under ``holdout_model``
    and ``model``; reports only consume the JSON-friendly keys.
    """
    config = config or TrainingConfig()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    label_names = tuple(str(name) for name in label_names)

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != y.size or y.shape != groups.shape:
        raise ValueError(
            "embeddings, y and groups disagree on the number of samples: "
            f"{embeddings.shape[0]}, {y.shape}, {groups.shape}"
        )
    if np.unique(y).size < 2:
        raise ValueError("need at least two classes to train a classifier")

    labels = np.arange(len(label_names), dtype=int)
    for value in np.unique(y):
        if int(value) not in labels:
            raise ValueError(
                f"label {value} is outside label_names {label_names}"
            )

    split = group_holdout(
        groups, y, test_size=config.test_size, seed=config.seed
    )
    train_index, test_index = split.train, split.test
    train_classes = np.unique(y[train_index])
    if train_classes.size < 2:
        raise ValueError(
            "the held-out split left a single class in the training part; "
            "add recordings or use group_by='state_block'"
        )

    cv, n_splits = inner_cv(
        groups[train_index],
        y[train_index],
        n_splits=config.n_splits,
        seed=config.seed,
    )
    study = run_study(
        embeddings[train_index],
        y[train_index],
        groups[train_index],
        config,
        labels=labels,
        cv=cv,
        n_splits=n_splits,
    )
    best_params = normalize_params(study.best_trial.params)
    best_folds = list(
        study.best_trial.user_attrs.get("fold_scores", [study.best_value])
    )

    log.info(
        "Best trial %d: %s -> %s=%.3f",
        study.best_trial.number,
        best_params,
        config.metric,
        study.best_value,
    )

    # Honest estimate: fit on the training groups, score once on held-out groups.
    holdout_model = _fit(
        best_params, embeddings[train_index], y[train_index], config
    )
    holdout_pred = holdout_model.predict(embeddings[test_index])
    holdout_proba = (
        align_probabilities(holdout_model, embeddings[test_index], labels)
        if config.probability
        else None
    )
    holdout_metrics = classification_metrics(
        y[test_index],
        holdout_pred,
        labels=labels,
        label_names=label_names,
        y_proba=holdout_proba,
        metric=config.metric,
    )

    # Out-of-fold predictions on the training groups for diagnostics.
    oof_pred = np.full(y[train_index].shape, -1, dtype=int)
    for fold_train, fold_test in cv.split(
        embeddings[train_index], y[train_index], groups=groups[train_index]
    ):
        fold_model = make_estimator(
            best_params, seed=config.seed, probability=False
        )
        fold_model.fit(
            embeddings[train_index][fold_train], y[train_index][fold_train]
        )
        oof_pred[fold_test] = fold_model.predict(
            embeddings[train_index][fold_test]
        )
    if np.any(oof_pred < 0):
        raise RuntimeError("cross-validation left samples without a prediction")
    oof_metrics = classification_metrics(
        y[train_index],
        oof_pred,
        labels=labels,
        label_names=label_names,
        metric=config.metric,
    )

    # Ship-ready model for the next step (saving happens in model_store).
    final_model = _fit(best_params, embeddings, y, config)

    return {
        "config": {
            key: str(value) if hasattr(value, "__fspath__") else value
            for key, value in asdict(config).items()
        },
        "label_names": list(label_names),
        "labels": labels.tolist(),
        "dataset_summary": dataset_summary,
        "embedding_meta": embedding_meta or {"shape": list(embeddings.shape)},
        "split": {
            "train_indices": train_index.tolist(),
            "test_indices": test_index.tolist(),
            "n_train": int(train_index.size),
            "n_test": int(test_index.size),
            "test_groups": list(split.test_groups),
            "n_train_groups": int(np.unique(groups[train_index]).size),
        },
        "cv": {
            "metric": config.metric,
            "n_splits": int(n_splits),
            "scores": [float(score) for score in best_folds],
            "mean": float(np.mean(best_folds)),
            "std": float(np.std(best_folds)),
        },
        "best_params": best_params,
        "best_value": float(study.best_value),
        "trials": trial_records(study),
        "holdout": {
            "indices": test_index.tolist(),
            "y_true": y[test_index],
            "y_pred": np.asarray(holdout_pred),
            "y_proba": holdout_proba,
            "metrics": holdout_metrics,
        },
        "oof": {
            "indices": train_index.tolist(),
            "y_true": y[train_index],
            "y_pred": oof_pred,
            "metrics": oof_metrics,
        },
        "model": final_model,
        "holdout_model": holdout_model,
        "embeddings": embeddings,
        "study": study,
    }


def train(
    dataset: Any,
    *,
    config: TrainingConfig | None = None,
    encoder: Any | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Embed an :class:`~hotb_2026.classification.dataset.EegDataset` and train.

    ``encoder`` defaults to a frozen :class:`~hotb_2026.features.ReveEncoder`.
    Pass a lightweight fake in tests, or precomputed embeddings to
    :func:`train_from_embeddings`.
    """
    config = config or TrainingConfig()
    if cache_dir is None:
        cache_dir = config.cache_dir
    embeddings, meta = embed_dataset(
        dataset,
        encoder=encoder,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return train_from_embeddings(
        embeddings,
        dataset.y,
        dataset.groups,
        config=config,
        label_names=dataset.label_names,
        dataset_summary=dataset.summary(),
        embedding_meta=meta,
    )


def trial_records(study: Any) -> list[dict[str, Any]]:
    """Flatten an Optuna study into rows that are easy to serialize."""
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        records.append(
            {
                "number": int(trial.number),
                "state": str(trial.state.name),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "fold_scores": list(trial.user_attrs.get("fold_scores", [])),
                "fold_std": trial.user_attrs.get("fold_std"),
            }
        )
    return records


__all__ = [
    "embed_dataset",
    "train",
    "train_from_embeddings",
    "trial_records",
]
